// One shared dashboard for native x64, x86 and compatibility routes.
// Tabs and expansion state belong to ImGui, not persisted inference settings.
#pragma once
#include "widgets.h"
#include "panel_sections.h"
#include "panel_timing.h"
#include "panel_chrome.h"
namespace phoenix_ui {
inline bool pipeline_group(Context& context, PipelineSection section) {
    return draw_pipeline_options(context.panel.pipeline, context.panel.model_library_directory,
        context.native_chain_available && !context.portable_backend_selected &&
            context.panel.execution_device == "native" && context.panel.mode != phoenix::nvidia,
        context.portable_chain_available,
        context.frame_services_available && context.final_frame_source && context.panel.mode != phoenix::nvidia,
        section,context.motion_services_available,context.panel.processing.enabled&&context.panel.processing.resize_method!=0,
        context.x86_transport);
}
// All exercises the real task renderers in CPU tests; it is not a visible tab.
inline bool draw_sections(Context& context, PanelPage page, bool advanced = false) {
    bool changed = false;
    auto before = context.panel;
    const bool all = page == PanelPage::All;
    if (page == PanelPage::Play || all) {
        if (all) changed |= draw_master_switch(context);
        changed |= section_card(all,"source_card","Source & model",
            "Choose the output. Processing devices and model files live in Setup.",blue,[&] {
            bool edited = draw_output_selection(context);
            edited |= draw_model_choice(context);
            if (!all && phoenix_ui::button("Models and runner setup")) context.requested_page = PanelPage::Setup;
            return edited;
        });
        changed |= section_card(all,"look_card","Look",
            "LIVE / Strength overrides apply while you play.",mint,[&] {
            const bool edited = draw_appearance_controls(context);
            if (!all) {
                help("Tone and structure keep their independent controls. Skin detail follows the automatic mask.");
                if (phoenix_ui::button("More image controls")) context.requested_page = PanelPage::Image;
            }
            return edited;
        });
        if (!all) changed |= section_card(false,"resolution_card","Resolution",
            "Normal uses the full canvas. Super resolution processes a smaller canvas, then reconstructs the output.",blue,[&] {
                return draw_resolution_controls(context);
            });
        changed |= section_card(all,"compare_card","Compare",
            "Inspect matching input and processed output; retained measurements are in Performance.",violet,[&] {
            bool edited = false;
            edited |= ImGui::Checkbox("Show on-screen labels", &context.panel.comparison_labels);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Show or hide the source labels over fullscreen and comparison output. Separator lines are controlled separately.");
            if (all || ImGui::CollapsingHeader("Compare and inspect")) {
                if (!all && context.panel.mode != phoenix::split && phoenix_ui::button("Compare input / processed")) {
                    const bool nvidia = context.panel.mode == phoenix::nvidia;
                    context.panel.mode = phoenix::split;
                    context.panel.comparison = 1;
                    context.panel.source_a = 0;
                    context.panel.source_b = nvidia ? 1 : 2;
                    edited = true;
                }
                edited |= draw_comparison_controls(context);
                edited |= pipeline_group(context,PipelineSection::Inspection);
            }
            ImGui::BeginDisabled(!context.screenshot_pair_available);
            if (phoenix_ui::button("Save original + modified")) context.action(Action::SaveScreenshotPair);
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("Save two PNGs from the same processed frame, before later game effects and the overlay. Direct game captures retain the selected comparison view; helper captures save the full processed result. HDR is converted to SDR.");
            if (!context.screenshot_pair_status.empty()) ImGui::TextWrapped("%s",context.screenshot_pair_status.c_str());
            if (!all) {
                if (phoenix_ui::button("Reset history")) context.action(Action::ResetHistory);
                if (phoenix_ui::button("Timing comparisons")) context.requested_page = PanelPage::Performance;
            }
            return edited;
        });
        changed |= section_card(all,"presets_card","Presets & quick actions",
            "Presets are reusable control snapshots. Save configuration stores the current setup.",blue,[&] {
                return draw_preset_controls(context);
            });
    }
    if (page == PanelPage::Image || all) {
        changed |= section_card(all,"image_card","Image controls",
            "Resolution, appearance and temporal behavior. Unsupported engine-guide options keep their availability restrictions.",violet,[&] {
            bool edited = false;
            ImGui::SeparatorText("Resolution and reconstruction");
            edited |= draw_resolution_controls(context);
            if (all || ImGui::CollapsingHeader("Motion and scene changes")) {
                edited |= pipeline_group(context,PipelineSection::Motion);
                if(!context.motion_status.empty())ImGui::TextWrapped("%s",context.motion_status.c_str());
            }
            if (all || ImGui::CollapsingHeader("Processing chain"))
                edited |= pipeline_group(context,PipelineSection::Chain);
            edited |= draw_advanced_appearance(context);
            edited |= draw_signal_controls(context);
            edited |= draw_guide_controls(context);
            edited |= draw_reference_reset(context);
            if (all || ImGui::CollapsingHeader("Temporal history and resets")) edited |= draw_history_controls(context);
            return edited;
        });
        if (context.final_frame_source)
            changed |= section_card(all,"depth_card","Depth",
                "Depth source and guide shaping for legacy games without engine depth.",mint,[&] {
                bool edited = false;
                constexpr std::array<int, 5> source_ids{0, 4, 2, 1, 3};
                const char* methods[]{"Auto", "Game depth", "NVIDIA NvDepthAnythingV2", "Depth Anything V2 Small", "None"};
                const char* captions[]{"Auto", "Game depth", "NVIDIA", "Small", "None"};
                const uint8_t enabled = uint8_t(context.legacy_depth_available);
                const uint8_t available[]{enabled, uint8_t(context.legacy_game_depth_available), enabled, enabled, enabled};
                auto& pipeline = context.panel.pipeline;
                const auto selected = std::find(source_ids.begin(), source_ids.end(), pipeline.legacy_depth);
                int choice = selected == source_ids.end() ? 0 : int(selected - source_ids.begin());
                if (choice_buttons("Depth source", &choice, methods, 5, captions, available)) {
                    pipeline.legacy_depth = source_ids.at(choice);
                    edited = true;
                }
                ImGui::BeginDisabled(!context.legacy_depth_available);
                edited |= ImGui::Checkbox("Half-resolution capture", &pipeline.legacy_depth_half_resolution);
                phoenix_ui::item_help("Capture and send the depth guide at half resolution in each axis; the runtime resamples it onto the model grid. Much cheaper to read back; depth edges get blockier.");
                edited |= ImGui::Checkbox("Validate depth", &pipeline.legacy_depth_validate);
                phoenix_ui::item_help("Reject captured depth that is non-finite, outside [0,1], or flat before sending it. The helper validates again before accepting game depth either way.");
                ImGui::EndDisabled();
                ImGui::BeginDisabled(!context.legacy_depth_available || pipeline.legacy_depth == 3);
                edited |= phoenix_ui::slider_float("Depth strength", &pipeline.legacy_depth_strength,
                    0.f, phoenix_pipeline::Options::maximum_depth_strength,
                    phoenix_pipeline::Options{}.legacy_depth_strength, "%.3fx", ImGuiSliderFlags_AlwaysClamp);
                edited |= ImGui::Checkbox("Invert depth", &pipeline.legacy_depth_invert);
                phoenix_ui::item_help("Reverse near and far in the preview and in the guide sent to the model.");
                edited |= ImGui::Checkbox("Fixed depth range", &pipeline.legacy_depth_fixed_range);
                phoenix_ui::item_help("Select a percentage window within the observed depth range. 0% is the closest sample and 100% is the farthest.");
                if (pipeline.legacy_depth_fixed_range) {
                    constexpr phoenix_pipeline::Options defaults;
                    float near_percent = 100.f * pipeline.legacy_depth_near;
                    if (phoenix_ui::slider_float("Near depth", &near_percent,
                            0.f, 100.f, 100.f * defaults.legacy_depth_near, "%.2f%%", ImGuiSliderFlags_AlwaysClamp)) {
                        pipeline.legacy_depth_near = std::min(near_percent * .01f, pipeline.legacy_depth_far - .00002f);
                        edited = true;
                    }
                    float far_percent = 100.f * pipeline.legacy_depth_far;
                    if (phoenix_ui::slider_float("Far depth", &far_percent,
                            0.f, 100.f, 100.f * defaults.legacy_depth_far, "%.2f%%", ImGuiSliderFlags_AlwaysClamp)) {
                        pipeline.legacy_depth_far = std::max(far_percent * .01f, pipeline.legacy_depth_near + .00002f);
                        edited = true;
                    }
                }
                ImGui::EndDisabled();
                ImGui::BeginDisabled(!context.legacy_depth_available);
                edited |= ImGui::Checkbox("Depth only", &context.panel.pipeline.legacy_depth_preview);
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip("Show the selected depth source in grayscale while neural rendering is enabled. Near surfaces are bright. Game depth shows black when the current game buffer is unavailable.");
                if (!context.legacy_depth_available)
                    help("Available for Phoenix and NVIDIA DLSS5 on the external mono CPU capture helper.");
                else if (!context.legacy_depth_status.empty())
                    ImGui::TextWrapped("%s", context.legacy_depth_status.c_str());
                return edited;
            });
        if (context.coherent_frames)
            changed |= section_card(all,"playback_card","Playback stability",
                "Keep the game from alternating between raw input and processed output.",blue,[&] {
                    return draw_playback_stability(context);
                });
    }
    if (page == PanelPage::Performance || all) {
        changed |= section_card(all,"quality_card","Quality & execution budget",
            "Blocks change the learned graph. Passes reprocess the previous output with separate history. FPS targets are not guarantees.",amber,[&] {
            bool edited = false;
            if (all || ImGui::CollapsingHeader("Pipeline responsiveness / output age"))
                edited |= pipeline_group(context,PipelineSection::Responsiveness);
            edited |= draw_network_passes(context);
            edited |= draw_neural_budget(context);
            return edited;
        });
        if (!all) {
            PanelCard card("timings_card","Retained timing comparisons",
                "Rows survive route and runner changes. GPU intervals, worker wall time and game FPS measure different work.",blue);
            if (card) {
                draw_timing(context);
                if (context.performance_details && (advanced || ImGui::CollapsingHeader("Execution details")))
                    context.performance_details();
            }
        }
    }
    if (page == PanelPage::Setup || all) {
        changed |= section_card(all,"models_card","Models & files",
            "Choose a local source, browse for its files and validate the selected model before use.",violet,[&] {
                if (all || ImGui::CollapsingHeader("Model files and validation",ImGuiTreeNodeFlags_DefaultOpen))
                    return draw_model_locations(context);
                return false;
            });
        changed |= section_card(all,"devices_card","Device & compatibility",
            "Phoenix runner choices apply to Phoenix output. NVIDIA DLL output follows the rendering GPU.",blue,[&] {
            bool edited = false;
            if (all || ImGui::CollapsingHeader("Processing device")) edited |= draw_device_controls(context);
            if (!all && context.setup_details) context.setup_details();
            if (all || ImGui::CollapsingHeader("Recovery")) edited |= pipeline_group(context,PipelineSection::Recovery);
            return edited;
        });
        changed |= section_card(all,"diagnostics_card","Diagnostics",
            "Captures, logging and scoped resets.",amber,[&] {
            bool edited = false;
            if (all || advanced || ImGui::CollapsingHeader("Diagnostics")) {
                edited |= draw_neural_diagnostics(context);
                edited |= draw_diagnostic_controls(context);
                if (!context.status.empty()) ImGui::TextWrapped("%s",context.status.c_str());
            }
            return edited;
        });
    }
    if (page == PanelPage::Hotkeys || all)
        changed |= section_card(all,"hotkeys_card","Hotkeys",
            "Choose keyboard shortcuts for Phoenix actions.",blue,[&] {
                return draw_hotkey_controls(context,!all);
            });
    context.live_selection_changed = phoenix::copy_live_selection(before,context.panel);
    context.live_tuning_changed = phoenix::copy_live_tuning(before,context.panel);
    return changed;
}
inline bool draw_controls(Context& context) {
    ScopedPanelStyle style;
    ImGui::PushID("phoenix_dashboard");
    const auto origin = ImGui::GetCursorScreenPos();
    const auto available = ImGui::GetContentRegionAvail();
    ImGui::GetWindowDrawList()->AddRectFilled(origin,
        ImVec2(origin.x + available.x, origin.y + available.y),ink(0x181b20),8);
    auto* storage = ImGui::GetStateStorage();
    // Runtime adapters recreate Context every frame. Keep deferred tab
    // navigation in ImGui's window storage, alongside expansion state.
    const auto navigation_id = ImGui::GetID("requested_page");
    std::optional<PanelPage> requested_page = context.requested_page;
    const int stored_page = storage->GetInt(navigation_id,-1);
    if (!requested_page && stored_page >= 0 && stored_page < int(PanelPage::All)) requested_page = PanelPage(stored_page);
    storage->SetInt(navigation_id,-1);
    context.requested_page.reset();
    const auto advanced_id = ImGui::GetID("advanced");
    bool advanced = storage->GetBool(advanced_id, false);
    bool changed = draw_header(context, advanced);
    storage->SetBool(advanced_id, advanced);
    // Anchor both regions to the host's available rectangle, not the previous
    // tab's content height (ImGui retains that height across tab switches).
    const auto content_origin = ImGui::GetCursorScreenPos();
    const float content_bottom = content_origin.y + ImGui::GetContentRegionAvail().y;
    if (ImGui::BeginTabBar("phoenix_pages", ImGuiTabBarFlags_FittingPolicyScroll)) {
        const char* names[]{"Play", "Image", "Performance", "Setup", "Hotkeys"};
        for (int index = 0; index < int(PanelPage::All); ++index) {
            const auto flags = requested_page && *requested_page == PanelPage(index) ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
            if (!ImGui::BeginTabItem(names[index],nullptr,flags)) continue;
            const float height = std::max(1.f, content_bottom - ImGui::GetCursorScreenPos().y);
            if (ImGui::BeginChild("controls", ImVec2(0, height), ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_HorizontalScrollbar)) {
                ImGui::PushItemWidth(std::max(110.f, ImGui::GetContentRegionAvail().x * .52f));
                changed |= draw_sections(context, PanelPage(index), advanced);
                ImGui::PopItemWidth();
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    if (context.requested_page) {
        storage->SetInt(navigation_id,int(*context.requested_page));
        context.requested_page.reset();
    }
    ImGui::PopID();
    return changed;
}
inline bool draw_compact(Context& context) {
    ScopedPanelStyle style(false);
    ImGui::PushID("phoenix_compact");
    auto before = context.panel;
    bool changed = draw_master_switch(context);
    changed |= draw_output_selection(context);
    changed |= draw_comparison_exit(context);
    changed |= draw_model_choice(context);
    changed |= draw_appearance_controls(context);
    if (!context.status.empty()) ImGui::TextWrapped("%s", context.status.c_str());
    draw_metrics(context);
    context.live_selection_changed = phoenix::copy_live_selection(before, context.panel);
    context.live_tuning_changed = phoenix::copy_live_tuning(before, context.panel);
    if (!context.error.empty()) ImGui::TextWrapped("%s", context.error.c_str());
    ImGui::PopID();
    return changed;
}
} // namespace phoenix_ui
