// One shared dashboard for native x64, x86 and compatibility routes.
// Tabs and expansion state belong to ImGui, not persisted inference settings.
#pragma once
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
        section);
}
// All exercises the real task renderers in CPU tests; it is not a fifth tab.
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
            if (!all && ImGui::Button("Models and runner setup")) context.requested_page = PanelPage::Setup;
            return edited;
        });
        changed |= section_card(all,"look_card","Look",
            "LIVE / Strength overrides apply while you play.",mint,[&] {
            const bool edited = draw_appearance_controls(context);
            if (!all) {
                help("Tone and structure keep their independent controls. Skin detail follows the automatic mask.");
                if (ImGui::Button("More image controls")) context.requested_page = PanelPage::Image;
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
                if (!all && context.panel.mode != phoenix::split && ImGui::Button("Compare input / processed")) {
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
            if (ImGui::Button("Save original + modified")) context.action(Action::SaveScreenshotPair);
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("Save two PNGs from the same processed frame, before later game effects and the overlay. Direct game captures retain the selected comparison view; helper captures save the full processed result. HDR is converted to SDR.");
            if (!context.screenshot_pair_status.empty()) ImGui::TextWrapped("%s",context.screenshot_pair_status.c_str());
            if (!all) {
                if (ImGui::Button("Reset history")) context.action(Action::ResetHistory);
                if (ImGui::Button("Timing comparisons")) context.requested_page = PanelPage::Performance;
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
            if (all || ImGui::CollapsingHeader("Motion and scene changes"))
                edited |= pipeline_group(context,PipelineSection::Motion);
            if (all || ImGui::CollapsingHeader("Processing chain"))
                edited |= pipeline_group(context,PipelineSection::Chain);
            edited |= draw_advanced_appearance(context);
            if (all || advanced || ImGui::CollapsingHeader("Advanced image controls")) {
                edited |= draw_signal_controls(context);
                edited |= draw_guide_controls(context);
                edited |= draw_reference_reset(context);
            }
            if (all || ImGui::CollapsingHeader("Temporal history and resets")) edited |= draw_history_controls(context);
            return edited;
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
        changed |= section_card(all,"diagnostics_card","Shortcuts & details",
            "Hotkeys, diagnostics and scoped resets remain available here.",amber,[&] {
            bool edited = draw_hotkey_controls(context);
            if (all || advanced || ImGui::CollapsingHeader("Diagnostics")) {
                edited |= draw_neural_diagnostics(context);
                edited |= draw_diagnostic_controls(context);
                if (!context.status.empty()) ImGui::TextWrapped("%s",context.status.c_str());
            }
            return edited;
        });
    }
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
    if (!requested_page && stored_page >= 0 && stored_page < 4) requested_page = PanelPage(stored_page);
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
    const float footer = ImGui::GetContentRegionAvail().x >= 420.f
        ? ImGui::GetFrameHeightWithSpacing() * 1.6f : ImGui::GetFrameHeightWithSpacing() * 3.8f;
    if (ImGui::BeginTabBar("phoenix_pages", ImGuiTabBarFlags_FittingPolicyScroll)) {
        const char* names[]{"Play", "Image", "Performance", "Setup"};
        for (int index = 0; index < 4; ++index) {
            const auto flags = requested_page && *requested_page == PanelPage(index) ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
            if (!ImGui::BeginTabItem(names[index],nullptr,flags)) continue;
            const float height = std::max(1.f, content_bottom - footer - ImGui::GetCursorScreenPos().y);
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
    ImGui::SetCursorScreenPos(ImVec2(content_origin.x, content_bottom - footer));
    if (ImGui::BeginChild("metrics", ImVec2(0, footer), ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar))
        draw_metrics(context);
    ImGui::EndChild();
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
