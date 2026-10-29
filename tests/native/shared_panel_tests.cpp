// Render the real shared widgets in an ImGui context, without a game or GPU.
// Link matching external ImGui 1.92.5 sources for this optional UI test only.
#ifndef IMGUI_DISABLE_INCLUDE_IMCONFIG_H
#define IMGUI_DISABLE_INCLUDE_IMCONFIG_H
#endif
#include "../../src/integration/reshade/shared_panel.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
ImRect clear_timing_button, confirm_timing_button;
struct Widget {
    ImGuiID id;
    std::string label;
    ImRect rect;
    bool disabled = false;
};
std::vector<Widget> widgets;
bool record_widgets = false;
struct Changes {
    bool changed = false, processing = false, selection = false, tuning = false;
    void merge(const Changes &other) {
        changed |= other.changed;
        processing |= other.processing;
        selection |= other.selection;
        tuning |= other.tuning;
    }
};
}
// Minimal test-engine hooks identify real widget geometry. The test then feeds
// ordinary ImGui events; production code needs no testing callbacks or hooks.
void ImGuiTestEngineHook_ItemAdd(ImGuiContext *context, ImGuiID id, const ImRect &rect,
                                 const ImGuiLastItemData *data) {
    if (!record_widgets || !id || !context->CurrentWindow || context->CurrentWindow->IDStack.empty())
        return;
    // BeginCombo has ItemAdd geometry but no ItemInfo label in ImGui 1.92.5.
    std::string label;
    for (const char *combo : {"Output", "View", "Source A", "Source B", "Depth", "Resolution mode", "Phoenix style", "Neural model", "Shared native models", "NVIDIA neural model", "Hook method", "Present DLSS guide reuse", "Scaling domain", "Downsample filter", "Upsample filter", "Scale reconstruction", "Input encoding", "NR signal encoding", "Use exposure value", "UI correction policy", "Target"})
        if (context->CurrentWindow->GetID(combo) == id)
            label = combo;
    const auto found = std::find_if(widgets.begin(), widgets.end(),
                                   [&](const Widget &widget) { return widget.id == id; });
    if (found == widgets.end())
        widgets.push_back({id, label, rect,
                           ((data ? data->ItemFlags : context->CurrentItemFlags) & ImGuiItemFlags_Disabled) != 0});
    else {
        // Tabs report their label before ItemAdd; replace that preliminary
        // rectangle with the real hit bounds when they arrive.
        found->rect = rect;
        found->disabled = ((data ? data->ItemFlags : context->CurrentItemFlags) & ImGuiItemFlags_Disabled) != 0;
        if (found->label.empty()) found->label = label;
    }
}
void ImGuiTestEngineHook_ItemInfo(ImGuiContext *context, ImGuiID id, const char *label,
                                  ImGuiItemStatusFlags) {
    if (std::strcmp(label, "Clear GPU timing") == 0)
        clear_timing_button = context->LastItemData.Rect;
    if (std::strcmp(label, "Clear timings") == 0) confirm_timing_button = context->LastItemData.Rect;
    if (record_widgets) {
        const auto found = std::find_if(widgets.begin(), widgets.end(),
                                       [&](const Widget &widget) { return widget.id == id; });
        if (found != widgets.end()) {
            found->label = label;
            found->disabled = (context->LastItemData.ItemFlags & ImGuiItemFlags_Disabled) != 0;
        } else
            widgets.push_back({id, label, context->LastItemData.Rect,
                               (context->LastItemData.ItemFlags & ImGuiItemFlags_Disabled) != 0});
    }
}
void ImGuiTestEngineHook_Log(ImGuiContext *, const char *, ...) {}
const char *ImGuiTestEngine_FindItemDebugLabel(ImGuiContext *, ImGuiID) { return ""; }
#include "dashboard_tests.h"
int run_tests() {
    struct LibraryFixture {
        std::filesystem::path root = std::filesystem::absolute(std::filesystem::temp_directory_path() /
            ("phoenix-shared-widget-models-" + std::to_string(GetCurrentProcessId())));
        LibraryFixture() {
            nlohmann::json entries = nlohmann::json::array();
            for (const auto* id : {"original", "small-fp8"}) {
                const auto folder = root / id;
                std::filesystem::create_directories(folder);
                std::ofstream(folder / "weights.bin", std::ios::binary) << "1234";
                const auto manifest = folder / "model.json";
                std::ofstream(manifest) << nlohmann::json({{"schema", "dlss5-phoenix-model/1"},
                    {"model", id}, {"architecture", phoenix_model::definition(id).architecture},
                    {"resource", {{"path", "weights.bin"}, {"bytes", 4}, {"sha256", std::string(64, 'a')}}}}).dump();
                entries.push_back({{"id", id}, {"label", std::string(id) == "original" ? "Original native export" : "Lightweight native export"},
                    {"model", id}, {"manifest", std::string(id) + "/model.json"},
                    {"manifest_sha256", phoenix_model::mapped_file_digest(manifest)}});
            }
            std::ofstream(root / "model-library.json") << nlohmann::json({{"schema", "phoenix-model-library/1"}, {"models", entries}}).dump();
        }
        ~LibraryFixture() { std::error_code error; std::filesystem::remove_all(root, error); }
    } model_library;
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
    std::array<int, 7> hotkeys{};
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
                     {"History policy", "Phoenix DLSS5 experiments",
                      "Transformer block repetitions", "Hotkeys", "Settings / logging",
                      "Phoenix DLSS5 model", "GPU stage breakdown", "Skin and automatic mask", "Appearance details", "Engine motion and depth"})
                    ImGui::GetStateStorage()->SetInt(ImGui::GetID(section), 1);
                phoenix_ui::draw_sections(context, phoenix_ui::PanelPage::All);
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
    check(phoenix_ui::capture_binding(hotkeys, 7, 0x75) == BindingResult::Invalid);
    check(phoenix_ui::capture_binding(hotkeys, 0, 256) == BindingResult::Invalid);
    check(phoenix_ui::capture_binding(hotkeys, 0, 0x75) == BindingResult::Accepted);
    check(hotkeys[0] == 0x75);
    check(phoenix_ui::capture_binding(hotkeys, 6, 0x74) == BindingResult::Accepted);
    check(hotkeys[6] == 0x74);
    check(phoenix_ui::capture_binding(hotkeys, 0, 0x74) == BindingResult::Duplicate);
    hotkeys[6]=0;
    check(phoenix_ui::valid_action(7) && !phoenix_ui::valid_action(8));
    std::array<int, 7> imported{{0x78, 0x78, 0x24, -1, 256, 0x1b}};
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
        if (ImGui::GetCurrentContext()->OpenPopupStack.empty()) ImGui::SetNextWindowFocus();
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
    check(actions == 0);
    draw_interaction();
    io.AddMousePosEvent(confirm_timing_button.GetCenter().x, confirm_timing_button.GetCenter().y);
    io.AddMouseButtonEvent(0, true); draw_interaction();
    io.AddMouseButtonEvent(0, false); draw_interaction();
    check(actions == 1);
    ImGui::NewFrame();
    ImGui::Begin("Shared compact panel checks");
    check(!phoenix_ui::draw_compact(interaction));
    ImGui::End();
    ImGui::Render();

    unsigned widget_checks = 0, backend_selections = 0, source_selections = 0,
             master_toggles = 0, strength_scenarios = 0, appearance_scenarios = 0;
    const auto require_widget = [&](bool condition, const std::string &message) {
        ++widget_checks;
        if (!condition)
            throw std::runtime_error("Shared UI widget contract: " + message);
    };
    const auto count_widget = [&](const char *label) {
        return std::count_if(widgets.begin(), widgets.end(), [&](const Widget &widget) {
            return widget.label == label;
        });
    };
    const auto widget_rect = [&](const char *label) {
        require_widget(count_widget(label) == 1,
                       std::string("Expected exactly one widget: ") + label + " (found " +
                           std::to_string(count_widget(label)) + ", mode " + std::to_string(settings.mode) +
                           ", appearance " + std::to_string(settings.controls.appearance) + ")");
        const auto found = std::find_if(widgets.begin(), widgets.end(), [&](const Widget &widget) {
            return widget.label == label;
        });
        require_widget(found->rect.GetWidth() > 0 && found->rect.GetHeight() > 0,
                       std::string("Widget has no clickable bounds: ") + label);
        return found->rect;
    };
    const auto widget_disabled = [&](const char *label) {
        widget_rect(label);
        return std::find_if(widgets.begin(), widgets.end(), [&](const Widget &widget) {
            return widget.label == label;
        })->disabled;
    };
    const auto callbacks_before = std::array<unsigned, 3>{persisted, applied, actions};
    for (bool compact : {false, true})
        for (bool frontend_master : {false, true}) {
            settings = {};
            settings.mode = phoenix_ui::phoenix::ours;
            settings.comparison = 0;
            settings.source_a = 0;
            settings.source_b = 2;
            settings.controls.enabled = true;
            settings.controls.tone = .65f;
            settings.intensity = 1.25f;
            settings.transformer_blocks = 5;
            settings.model_dll = "pending-model.dll";
            bool enabled = true;
            interaction.processing_enabled = frontend_master ? &enabled : nullptr;
            interaction.direct_nr_ready = true;
            interaction.comparison_views = true;
            interaction.direct_game = true;
            interaction.status = "Ready";
            const std::string window_name = std::string("Shared widget interactions ") +
                (compact ? "compact " : "full ") + (frontend_master ? "frontend" : "direct");
            const auto frame = [&] {
                widgets.clear();
                ImGui::NewFrame();
                ImGui::SetNextWindowPos({0, 0});
                ImGui::SetNextWindowSize({1600, 6000});
                ImGui::Begin(window_name.c_str());
                if (!compact)
                    for (const char* group : {"Skin and automatic mask", "Appearance details", "Engine motion and depth"})
                        ImGui::GetStateStorage()->SetInt(ImGui::GetID(group), 1);
                record_widgets = true;
                const bool changed = compact ? phoenix_ui::draw_compact(interaction)
                                             : phoenix_ui::draw_sections(interaction, phoenix_ui::PanelPage::All);
                record_widgets = false;
                ImGui::End();
                ImGui::Render();
                return Changes{changed, interaction.processing_changed,
                               interaction.live_selection_changed, interaction.live_tuning_changed};
            };
            const auto click_at = [&](ImVec2 point) {
                Changes result;
                io.AddMousePosEvent(point.x, point.y);
                result.merge(frame());
                io.AddMouseButtonEvent(0, true);
                result.merge(frame());
                io.AddMouseButtonEvent(0, false);
                result.merge(frame());
                return result;
            };
            const auto click = [&](const char *label) {
                return click_at(widget_rect(label).GetCenter());
            };
            const auto select = [&](const char *combo, const char *option) {
                frame(); frame(); // Settle text wrapping after the preceding selection.
                const auto rect = widget_rect(combo);
                // Combo bounds include the label; click inside the frame at its left edge.
                Changes result = click_at(ImVec2(rect.Min.x + 12, rect.GetCenter().y));
                result.merge(frame()); // Let an appearing popup settle before locating its rows.
                require_widget(ImGui::GetCurrentContext()->OpenPopupStack.Size > 0,
                               std::string("Combo did not open: ") + combo + " at " + std::to_string(rect.Min.x) + "," + std::to_string(rect.Min.y) + " hover=" + (ImGui::GetCurrentContext()->HoveredWindow ? ImGui::GetCurrentContext()->HoveredWindow->Name : "none"));
                result.merge(click(option));
                return result;
            };
            const auto expect_selection = [&](Changes result, const char *message) {
                require_widget(result.changed && result.selection && !result.processing && !result.tuning, message);
            };
            frame();
            require_widget(!frame().changed, "Passive drawing changed settings");
            const char *backend = compact ? "Output" : "Output";
            require_widget(count_widget("Enable neural rendering") == 1,
                           "There must be one shared enable checkbox");
            for (const char *obsolete : {"Enable graphics bridge", "Graphics bridge", "Enable NR",
                                         "Neural rendering (master switch)"})
                require_widget(count_widget(obsolete) == 0,
                               std::string("Obsolete master control remains: ") + obsolete);
            require_widget(widget_rect("Enable neural rendering").Min.y < widget_rect(backend).Min.y,
                           "Enable checkbox must precede Backend");
            if (!compact) {
                require_widget(count_widget("NVIDIA DLL search folder") == 1,
                               "Phoenix DLSS5 hides the shared DLL folder setting");
                widget_rect("Browse...##nvidia_library");
                const auto saved_settings = settings;
                click("Phoenix DLSS5 experiments");
                for (const char* model : {"original", "small-fp8"}) {
                    settings.model_id = model;
                    settings.execution_device = "native";
                    settings.experimental = settings.automatic_budget = true;
                    frame();
                    require_widget(!widget_disabled("Transformer blocks") &&
                                       !widget_disabled("Adaptive blocks") &&
                                       !widget_disabled("Minimum blocks"),
                                   std::string(model) + " depth/budget controls unavailable");
                    if (!count_widget("Block 1 repetitions"))
                        click("Transformer block repetitions");
                    for (unsigned block = 0; block < 8; ++block) {
                        const auto label = "Block " + std::to_string(block + 1) + " repetitions";
                        require_widget(!widget_disabled(label.c_str()),
                                       std::string(model) + " per-block control disabled");
                        settings.block_repeats[block] = 4;
                        frame();
                        const auto rect = widget_rect(label.c_str());
                        require_widget(click_at({rect.Min.x + 5, rect.GetCenter().y}).changed &&
                                           settings.block_repeats[block] == 0,
                                       std::string(model) + " block skip did not update settings");
                    }
                    select("Target", "Desired game FPS"); frame();
                    require_widget(settings.budget_mode == 1 && !widget_disabled("Desired FPS"),
                                   std::string(model) + " FPS target unavailable");
                    select("Target", "Phoenix DLSS5 GPU time"); frame();
                    require_widget(settings.budget_mode == 0 &&
                                       !widget_disabled("Phoenix DLSS5 GPU budget (ms)"),
                                   std::string(model) + " GPU-time budget unavailable");
                    require_widget(settings.valid(), "Granular model controls corrupted settings");
                }
                click("Phoenix DLSS5 experiments");
                settings = saved_settings;
                frame();
                click("Processing pipeline"); frame();
                require_widget(widget_disabled("Resolution scale"), "Legacy bridge must disable explicit scale");
                const auto enabled_pipeline=click("Explicit scaling and color controls");frame();
                require_widget(enabled_pipeline.changed && settings.processing.enabled,
                               "Processing checkbox must update the shared runtime setting");
                require_widget(!widget_disabled("Resolution scale"),"Explicit scaling must become editable");
                select("Hook method","Present");
                require_widget(settings.processing.hook==3,"Present hook selection");
                select("Present DLSS guide reuse","Required");
                require_widget(settings.processing.reuse_guides==2,"Required guide policy selection");
                select("Input encoding","HDR10");
                require_widget(settings.processing.encoding==3,"Explicit HDR10 input selection");
                select("NR signal encoding","Linear SDR");
                require_widget(settings.processing.signal==2,"Explicit linear NR signal selection");
                for(const char* label:{"Scaling domain","Downsample filter","Upsample filter",
                    "Scale reconstruction","Override diffuse white","Exposure multiplier",
                    "Use exposure value","Block Hudless","UI correction policy","Network passes per frame"})
                    require_widget(count_widget(label)==1,std::string("Missing processing widget: ")+label);
                // Restore this scenario so existing presentation/tuning checks
                // continue to exercise their original, independent defaults.
                settings.processing={};click("Processing pipeline");frame();
            }
            const auto pending = settings;
            const auto pending_unchanged = [&] {
                require_widget(settings.controls.tone == pending.controls.tone &&
                                   settings.controls.enabled == pending.controls.enabled &&
                                   settings.intensity == pending.intensity &&
                                   settings.transformer_blocks == pending.transformer_blocks &&
                                   settings.model_dll == pending.model_dll,
                               "Live selection or enable toggle overwrote pending tuning");
            };
            const auto toggle_master = [&] {
                const bool before = frontend_master ? enabled : settings.uplift;
                const bool legacy_before = settings.uplift;
                const int mode_before = settings.mode;
                const auto result = click("Enable neural rendering");
                require_widget(result.changed && result.processing && !result.selection && !result.tuning,
                               "Master toggle must report only a processing change");
                require_widget((frontend_master ? enabled : settings.uplift) == !before &&
                                   settings.mode == mode_before,
                               "Master toggle must change enable independently of Backend");
                if (frontend_master)
                    require_widget(settings.uplift == legacy_before,
                                   "Frontend enable checkbox must use its supplied bool");
                pending_unchanged();
                ++master_toggles;
            };
            toggle_master();
            if (!compact) {
                // Labels must also be reachable without switching to a split view.
                require_widget(settings.mode != phoenix_ui::phoenix::split,
                               "Fullscreen label test must start outside Comparison");
                const bool labels_before = settings.comparison_labels;
                expect_selection(click("Show on-screen labels"),
                                 "Fullscreen labels must report only live selection");
                require_widget(settings.comparison_labels != labels_before,
                               "Fullscreen label toggle failed");
                click("Show on-screen labels");
            }
            expect_selection(select(backend, "Comparison"), "Comparison must report live selection");
            require_widget(settings.mode == phoenix_ui::phoenix::split,
                           "Phoenix DLSS5 -> Comparison selection failed");
            require_widget(settings.comparison == 1,
                           "Entering Comparison must normalize a legacy three-way view");
            require_widget(!(frontend_master ? enabled : settings.uplift),
                           "Selecting Backend while disabled enabled processing");
            ++backend_selections;
            pending_unchanged();
            toggle_master();
            if (!compact) {
                expect_selection(select("Source A", "NVIDIA DLSS5"), "Source A must report live selection");
                require_widget(settings.source_a == 1 && settings.source_b == 2,
                               "Source A selection changed the wrong source");
                expect_selection(select("Source B", "Input"), "Source B must report live selection");
                require_widget(settings.source_a == 1 && settings.source_b == 0,
                               "Source B selection changed the wrong source");
                expect_selection(click("Swap A / B"), "Swap must report live selection");
                require_widget(settings.source_a == 0 && settings.source_b == 1,
                               "Swap did not exchange comparison sources");
                expect_selection(select("Source B", "Phoenix DLSS5"),
                                 "Phoenix DLSS5 source must report live selection");
                require_widget(settings.source_a == 0 && settings.source_b == 2,
                               "Input/Phoenix DLSS5 pair was not restored");
                source_selections += 4;
                const float divider_before = settings.divider;
                const auto divider = widget_rect("Split position");
                expect_selection(click_at(ImVec2(divider.Min.x + 30, divider.GetCenter().y)),
                                 "Divider must report live selection");
                require_widget(settings.divider != divider_before, "Divider click changed no value");
                const bool lines_before = settings.comparison_lines;
                expect_selection(click("Separator lines"), "Separator lines must report live selection");
                require_widget(settings.comparison_lines != lines_before, "Separator toggle failed");
                const bool labels_before = settings.comparison_labels;
                expect_selection(click("Show on-screen labels"), "Labels must report live selection");
                require_widget(settings.comparison_labels != labels_before, "Labels toggle failed");
                expect_selection(select("View", "Absolute RGB"), "View must report live selection");
                require_widget(settings.comparison == 2, "Absolute RGB selection failed");
                require_widget(count_widget("Split position") == 0 &&
                                   count_widget("Separator lines") == 0 &&
                                   count_widget("Difference gain") == 1,
                               "Difference view did not update conditional controls");
                const float gain_before = settings.difference_gain;
                const auto gain = widget_rect("Difference gain");
                expect_selection(click_at(ImVec2(gain.Min.x + 80, gain.GetCenter().y)),
                                 "Difference gain must report live selection");
                require_widget(settings.difference_gain != gain_before, "Difference gain click failed");
                expect_selection(select("View", "Pair wipe"), "Pair wipe must report live selection");
                require_widget(settings.comparison == 1 && count_widget("Split position") == 1 &&
                                   count_widget("Difference gain") == 0,
                               "Pair wipe did not restore conditional controls");
                pending_unchanged();
            }
            expect_selection(select(backend, "NVIDIA DLSS5 DLL"), "NVIDIA DLSS5 Backend must report live selection");
            require_widget(settings.mode == phoenix_ui::phoenix::nvidia,
                           "Comparison -> NVIDIA DLSS5 selection failed");
            if (!compact)
                require_widget(count_widget("NVIDIA DLL search folder") == 1,
                               "NVIDIA DLSS5 hides the shared DLL folder setting");
            ++backend_selections;
            pending_unchanged();
            expect_selection(select(backend, "Input"), "Input Backend must report live selection");
            require_widget(settings.mode == phoenix_ui::phoenix::input,
                           "NVIDIA DLSS5 -> Input selection failed");
            ++backend_selections;
            require_widget(count_widget("Enable neural rendering") == 1,
                           "Input mode hid the master checkbox");
            toggle_master();
            toggle_master();
            expect_selection(select(backend, "Phoenix DLSS5"), "Phoenix DLSS5 Backend must report live selection");
            require_widget(settings.mode == phoenix_ui::phoenix::ours,
                           "Input -> Phoenix DLSS5 selection failed");
            ++backend_selections;
            pending_unchanged();
            if (!compact) {
                settings.model_library_directory = phoenix_model::source_path_string(model_library.root / "missing");
                frame(); frame();
                require_widget(count_widget("Shared native models") == 0, "Unavailable optional models must not be listed");
                settings.model_library_directory = phoenix_model::source_path_string(model_library.root);
                frame(); frame();
                require_widget(count_widget("Shared native models") == 1, "Installed shared models picker is missing");
                const auto shared_light = select("Shared native models", "Lightweight native export");
                require_widget(shared_light.changed && settings.model_id == "small-fp8" &&
                    settings.model_source == "model_folder" &&
                    std::filesystem::u8path(settings.model_folder) == model_library.root / "small-fp8",
                    "Shared selection must set the matching model and central path together");
                const auto shared_original = select("Shared native models", "Original native export");
                require_widget(shared_original.changed && settings.model_id == "original", "Shared Original model selection failed");
                settings.model_source = "configured"; settings.model_folder.clear(); settings.model_library_directory.clear();
                frame(); frame();
                const auto selected = select("Neural model", "Lightweight FP8");
                require_widget(selected.changed && !selected.processing && !selected.tuning &&
                                   settings.model_id == "small-fp8",
                               "Neural model dropdown did not select Lightweight FP8");
                const auto restored = select("Neural model", "Original");
                require_widget(restored.changed && settings.model_id == "original",
                               "Neural model dropdown did not restore Original");
                frame();
                frame(); // Retire the model popup before testing an unrelated tuning slider.
                const float tone_before = settings.controls.tone;
                const auto tone = widget_rect("Local tone");
                const auto result = click_at(ImVec2(tone.Min.x + 30, tone.GetCenter().y));
                require_widget(result.changed && !result.selection && !result.processing &&
                                   result.tuning && settings.controls.tone != tone_before,
                               "Tuning slider must report an immediate live change: changed=" + std::to_string(result.changed) +
                               " selection=" + std::to_string(result.selection) + " tuning=" + std::to_string(result.tuning) +
                               " before=" + std::to_string(tone_before) + " after=" + std::to_string(settings.controls.tone));
            }
            const auto idle = frame();
            require_widget(!idle.changed && !idle.processing && !idle.selection && !idle.tuning,
                           "Idle frame repeated change notifications");
            if (!compact && frontend_master) {
                const bool was_enabled = enabled;
                const auto result = click("Reset neural overrides");
                require_widget(result.changed && !result.processing && enabled == was_enabled && settings.uplift,
                               "Reset overrides must not disable compatibility processing");
            }
            if (!compact) {
                for (bool real_guides : {false, true})
                    for (int mode : {phoenix_ui::phoenix::nvidia, phoenix_ui::phoenix::split,
                                     phoenix_ui::phoenix::ours}) {
                        interaction.real_guides = real_guides;
                        settings.mode = mode;
                        settings.controls = {};
                        frame();
                        require_widget(!widget_disabled("Override strengths") &&
                                           !widget_disabled("Override skin and guides"),
                                       "Strength/skin override checkbox must not depend on engine guides");
                        auto changed = click("Override strengths");
                        require_widget(changed.changed && changed.tuning && !changed.selection && !changed.processing &&
                                           settings.controls.enabled,
                                       "Strength override did not enable as a live setting");
                        for (const char* label : {"Local tone", "Local structure"}) {
                            require_widget(!widget_disabled(label), "Strength slider remained disabled");
                            const auto rect = widget_rect(label);
                            require_widget(click_at({rect.Min.x + 30, rect.GetCenter().y}).changed,
                                           "Strength slider could not be edited");
                        }
                        require_widget(widget_disabled("Automatic skin mask"),
                                       "Skin controls must respect their override checkbox");
                        changed = click("Override skin and guides");
                        require_widget(changed.changed && changed.tuning && !changed.selection && !changed.processing &&
                                           settings.controls.advanced,
                                       "Skin override could not be enabled without engine guides");
                        require_widget(!widget_disabled("Automatic skin mask") && widget_disabled("Skin structure"),
                                       "Skin strength must wait for automatic masking, not engine guides");
                        require_widget(click("Automatic skin mask").changed && settings.controls.auto_mask,
                                       "Automatic skin mask checkbox did not work");
                        require_widget(!widget_disabled("Skin structure"), "Skin slider remained disabled");
                        const auto skin = widget_rect("Skin structure");
                        changed = click_at({skin.Min.x + 80, skin.GetCenter().y});
                        require_widget(changed.changed && changed.tuning && !changed.selection && !changed.processing &&
                                           settings.controls.skin != -1,
                                       "Skin slider did not edit a live setting");
                        // Keep the mouse held: strength edits must not wait for
                        // release or an Apply button to become live.
                        io.AddMousePosEvent(skin.Min.x + 140, skin.GetCenter().y);
                        io.AddMouseButtonEvent(0, true);
                        const auto dragging = frame();
                        require_widget(ImGui::IsAnyItemActive() && dragging.tuning,
                                       "Skin edits waited until the drag ended");
                        io.AddMouseButtonEvent(0, false);
                        frame();
                        require_widget(widget_disabled("Motion X scale") == !real_guides &&
                                           widget_disabled("Motion Y scale") == !real_guides &&
                                           widget_disabled("Depth") == !real_guides,
                                       "Only engine motion/depth controls should require real guides");
                        const auto motion = widget_rect("Motion X scale");
                        changed = click_at({motion.Min.x + 30, motion.GetCenter().y});
                        require_widget(changed.changed == real_guides &&
                                           (settings.controls.motion_x != 1) == real_guides,
                                       "Motion control ignored guide availability");
                        require_widget(click("R##s.live.skin").changed && settings.controls.skin == -1,
                                       "Skin reset did not restore follow-local-structure value");
                        require_widget(click("Automatic skin mask").changed && !settings.controls.auto_mask &&
                                           widget_disabled("Skin structure"),
                                       "Disabling automatic masking did not disable skin strength");
                        require_widget(click("Override skin and guides").changed && !settings.controls.advanced &&
                                           widget_disabled("Automatic skin mask") && widget_disabled("Motion X scale"),
                                       "Disabling advanced overrides did not disable its child controls");
                        require_widget(settings.valid(), "Strength/skin interaction corrupted settings");
                        ++strength_scenarios;
                    }
            }
            if (!compact) {
                interaction.portable_backbuffer = false;
                interaction.real_guides = false;
                for (int mode : {phoenix_ui::phoenix::ours, phoenix_ui::phoenix::split}) {
                    settings.mode = mode;
                    settings.controls = {};
                    const float nvidia_intensity = settings.intensity;
                    const int nvidia_style = settings.style;
                    frame(); frame();
                    require_widget(widget_disabled("Phoenix intensity") && widget_disabled("Phoenix style"),
                                   "Appearance children must start disabled");
                    const auto enabled_appearance = click("Phoenix appearance (experimental)");
                    require_widget(enabled_appearance.changed && enabled_appearance.tuning &&
                                       !enabled_appearance.selection && !enabled_appearance.processing &&
                                       settings.controls.appearance,
                                   "Appearance enable must apply as a live tuning edit");
                    frame();
                    require_widget(!widget_disabled("Phoenix intensity") && !widget_disabled("Phoenix style"),
                                   "Appearance controls must work without engine motion/depth");
                    const auto intensity = widget_rect("Phoenix intensity");
                    const auto edited = click_at({intensity.Min.x + 55, intensity.GetCenter().y});
                    require_widget(edited.changed && edited.tuning && settings.controls.intensity != 1.f,
                                   "Phoenix intensity slider did not edit the live override");
                    const auto natural = select("Phoenix style", "Natural");
                    require_widget(natural.changed && natural.tuning && settings.controls.style == 1,
                                   "Natural style selection did not edit the Phoenix override");
                    const auto cinematic = select("Phoenix style", "Cinematic");
                    require_widget(cinematic.changed && cinematic.tuning && settings.controls.style == 2,
                                   "Cinematic style selection did not edit the Phoenix override");
                    require_widget(settings.intensity == nvidia_intensity && settings.style == nvidia_style,
                                   "Phoenix appearance edited the separate NVIDIA overrides");
                    phoenix_addon::Controls host_controls;
                    host_controls.intensity = .3f;
                    host_controls.style = 1;
                    host_controls.ui_correction = 1;
                    const auto effective = settings.controls.apply(host_controls);
                    require_widget(effective.intensity == settings.controls.intensity && effective.style == 2 &&
                                       effective.ui_correction == 1 && host_controls.intensity == .3f,
                                   "Appearance values did not reach frame controls independently of host UI correction");
                    nlohmann::json saved = nlohmann::json::object();
                    phoenix_ui::phoenix::visit_live_tuning(settings, [&](const char* key, auto& value) {
                        saved[key] = value;
                    });
                    phoenix_ui::Settings restored;
                    phoenix_ui::phoenix::visit_live_tuning(restored, [&](const char* key, auto& value) {
                        saved.at(key).get_to(value);
                    });
                    require_widget(restored.controls.appearance &&
                                       restored.controls.intensity == settings.controls.intensity &&
                                       restored.controls.style == 2 && restored.intensity == nvidia_intensity &&
                                       restored.style == nvidia_style,
                                   "Persisted Phoenix tuning failed to round-trip independently of NVIDIA values");
                    const float retained_intensity = settings.controls.intensity;
                    require_widget(click("Phoenix appearance (experimental)").changed &&
                                       !settings.controls.appearance,
                                   "Appearance disable did not switch off the override");
                    frame();
                    require_widget(widget_disabled("Phoenix intensity") && widget_disabled("Phoenix style") &&
                                       settings.controls.intensity == retained_intensity && settings.controls.style == 2,
                                   "Appearance disable must retain the selected values");
                    const auto disabled = settings.controls.apply(host_controls);
                    require_widget(disabled.intensity == .3f && disabled.style == 1,
                                   "Disabled appearance did not restore the host values");
                    ++appearance_scenarios;
                }
                settings.mode = phoenix_ui::phoenix::nvidia;
                frame();
                require_widget(count_widget("Phoenix appearance (experimental)") == 0,
                               "NVIDIA-only mode exposed Phoenix appearance controls");
            }
            require_widget(std::array<unsigned, 3>{persisted, applied, actions} == callbacks_before,
                           "Shared controls dispatched a persistence/runtime callback");
            if (!compact) {
                settings = {};
                settings.mode = phoenix_ui::phoenix::ours;
                settings.experimental = true;
                settings.controls.enabled = true;
                interaction.portable_backbuffer = true;
                interaction.input_capture = false;
                frame(); frame();
                require_widget(widget_disabled("Resolution mode"), "Portable mode exposed unsupported super resolution");
                // Expand the real quality controls, preserving the production UI path.
                if (!count_widget("Adaptive blocks")) {
                    click("Phoenix DLSS5 experiments");
                    frame();
                }
                require_widget(widget_disabled("Adaptive blocks"), "Portable mode exposed unsupported adaptive blocks");
                interaction.portable_gpu_budget = true;
                interaction.fps_budget = false;
                settings.automatic_budget = true;
                for (const char* model : {"original", "small-fp8"}) {
                    settings.model_id = model;
                    frame();
                    require_widget(!widget_disabled("Adaptive blocks") &&
                                       !widget_disabled("Phoenix DLSS5 GPU budget (ms)") &&
                                       !widget_disabled("Minimum blocks"),
                                   std::string(model) + " measured portable budget controls unavailable");
                }
                interaction.portable_gpu_budget = false;
                require_widget(count_widget("NVIDIA DLSS5-only overrides") == 0,
                               "Portable Phoenix DLSS5 showed NVIDIA DLSS5-only controls");
                require_widget(count_widget("Phoenix DLSS5 processing device") == 0,
                               "Portable mode duplicated the selected-device control");
                require_widget(count_widget("Phoenix appearance (experimental)") == 0 &&
                                   count_widget("Phoenix intensity") == 0 && count_widget("Phoenix style") == 0,
                               "Portable final-color route exposed unsupported appearance controls");
                interaction.portable_backbuffer = false;
                interaction.fps_budget = true;
                interaction.input_capture = true;
            }
        }
    interaction.processing_enabled = nullptr;
    const auto dashboard = run_dashboard_tests(interaction);
    ImGui::DestroyContext();
    std::cout << nlohmann::json({{"passed", true}, {"dashboard", dashboard},
                                 {"bits", sizeof(void *) * 8},
                                 {"rendered_panels", rendered},
                                 {"validation_controls_drawn", validation_polls},
                                 {"runtime_actions_while_drawing", 0},
                                 {"policy_action_checks", policy_checks},
                                 {"real_widget_checks", widget_checks},
                                 {"backend_selections", backend_selections},
                                 {"comparison_source_selections", source_selections},
                                 {"master_toggles", master_toggles},
                                 {"strength_skin_scenarios", strength_scenarios},
                                 {"appearance_scenarios", appearance_scenarios},
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
