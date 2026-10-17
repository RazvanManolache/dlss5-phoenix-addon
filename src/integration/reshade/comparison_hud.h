#pragma once
#include <imgui.h>
#include "../../native/support/panel_visibility.h"
#include "../../native/support/comparison.h"
#include <algorithm>
#include <cstdio>
#include <string>

namespace phoenix_ui {
inline void draw_presentation_hud(const phoenix_addon::phoenix::Settings& presentation) {
    namespace phoenix = phoenix_addon::phoenix;
    auto* draw = ImGui::GetBackgroundDrawList();
    const ImVec2 origin(0, 0), size = ImGui::GetIO().DisplaySize;
    auto line = [&](float fraction) {
        const float horizontal_position = origin.x + size.x * fraction;
        draw->AddLine(ImVec2(horizontal_position, origin.y),
                      ImVec2(horizontal_position, origin.y + size.y), IM_COL32(0, 0, 0, 255), 4);
        draw->AddLine(ImVec2(horizontal_position, origin.y),
                      ImVec2(horizontal_position, origin.y + size.y), IM_COL32(255, 255, 255, 255),
                      1);
    };
    auto label = [&](float left, float right, const std::string& text) {
        auto extent = ImGui::CalcTextSize(text.c_str());
        const float scale =
            std::min(1.f, std::max(1.f, size.x * (right - left) - 14) / std::max(1.f, extent.x));
        extent.x *= scale;
        extent.y *= scale;
        const float horizontal_position = origin.x + size.x * (left + right) * .5f - extent.x * .5f,
                    y = origin.y + 48;
        draw->AddRectFilled(ImVec2(horizontal_position - 6, y - 4),
                            ImVec2(horizontal_position + extent.x + 6, y + extent.y + 4),
                            IM_COL32(0, 0, 0, 210), 3);
        draw->AddText(ImGui::GetFont(), ImGui::GetFontSize() * scale,
                      ImVec2(horizontal_position, y), IM_COL32(255, 255, 255, 255), text.c_str());
    };
    // Saved comparison presets survive Backend switches; furniture follows the displayed route.
    if (phoenix::PanelVisibility{presentation}.wipe_hud()) {
        if (presentation.comparison == 0) {
            if (presentation.comparison_lines) {
                line(presentation.divider * .5f);
                line(presentation.divider);
            }
            if (presentation.comparison_labels) {
                label(0, presentation.divider * .5f, "Input");
                label(presentation.divider * .5f, presentation.divider, "NVIDIA");
                label(presentation.divider, 1, "DLSS5 Phoenix");
            }
        } else if (presentation.comparison == 1) {
            if (presentation.comparison_lines)
                line(presentation.divider);
            if (presentation.comparison_labels) {
                label(0, presentation.divider, phoenix::source_names[presentation.source_a]);
                label(presentation.divider, 1, phoenix::source_names[presentation.source_b]);
            }
        } else if (presentation.comparison_labels) {
            char text[160]{};
            std::snprintf(text, sizeof(text), "%s - %s | %s | %.1fx",
                          phoenix::source_names[presentation.source_a],
                          phoenix::source_names[presentation.source_b],
                          phoenix::comparison_names[presentation.comparison],
                          presentation.difference_gain);
            label(0, 1, text);
        }
    }
    if (presentation.mode != phoenix::split && presentation.comparison_labels) {
        const char* name = presentation.mode == phoenix::nvidia  ? "NVIDIA"
                           : presentation.mode == phoenix::input ? "Input"
                                                                 : "DLSS5 Phoenix";
        label(0, 1, name);
    }
}
} // namespace phoenix_ui
