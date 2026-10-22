// Local visual language for the Phoenix panel. No renderer, font or asset files.
#pragma once
#include <imgui.h>
#include <algorithm>
namespace phoenix_ui {
inline ImVec4 rgb(unsigned hex, float alpha = 1.f) {
    return ImVec4(float((hex >> 16) & 255) / 255, float((hex >> 8) & 255) / 255,
                  float(hex & 255) / 255, alpha);
}
inline ImU32 ink(unsigned hex, float alpha = 1.f) { return ImGui::ColorConvertFloat4ToU32(rgb(hex, alpha)); }
inline constexpr unsigned blue = 0x86b8ff, mint = 0x8ed6b4, amber = 0xf0cb8b, violet = 0xb7a5ed;

struct ScopedPanelStyle {
    int colors = 0;
    void color(ImGuiCol slot, unsigned value) { ImGui::PushStyleColor(slot, rgb(value)); ++colors; }
    explicit ScopedPanelStyle(bool tinted = true) {
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5.f);
        ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 4.f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(9.f, 6.f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(9.f, 9.f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.f);
        ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.f, 10.f));
        ImGui::PushStyleVar(ImGuiStyleVar_TabRounding, 5.f);
        if (!tinted) return;
        color(ImGuiCol_Text, 0xecf0f5); color(ImGuiCol_TextDisabled, 0xadb8c8);
        color(ImGuiCol_ChildBg, 0x181b20); color(ImGuiCol_Border, 0x404a58);
        color(ImGuiCol_FrameBg, 0x303844); color(ImGuiCol_FrameBgHovered, 0x3b4b61);
        color(ImGuiCol_FrameBgActive, 0x405572); color(ImGuiCol_Button, 0x2d4057);
        color(ImGuiCol_ButtonHovered, 0x405c7d); color(ImGuiCol_ButtonActive, 0x4b719a);
        color(ImGuiCol_Header, 0x2c3645); color(ImGuiCol_HeaderHovered, 0x3c526c);
        color(ImGuiCol_HeaderActive, 0x365b84); color(ImGuiCol_CheckMark, blue);
        color(ImGuiCol_SliderGrab, blue); color(ImGuiCol_SliderGrabActive, 0xb6d8ff);
        color(ImGuiCol_Tab, 0x232b36); color(ImGuiCol_TabHovered, 0x3b536f);
        color(ImGuiCol_TabSelected, 0x365b84); color(ImGuiCol_TabSelectedOverline, blue);
        color(ImGuiCol_Separator, 0x404a58); color(ImGuiCol_PlotLines, mint);
        color(ImGuiCol_ScrollbarBg, 0x171d25); color(ImGuiCol_ScrollbarGrab, 0x445266);
        color(ImGuiCol_TableHeaderBg, 0x303844);
    }
    ~ScopedPanelStyle() { ImGui::PopStyleColor(colors); ImGui::PopStyleVar(8); }
    ScopedPanelStyle(const ScopedPanelStyle&) = delete;
};

inline void heading(const char* title, unsigned color = 0xecf0f5, float scale = 1.12f) {
    const auto position = ImGui::GetCursorScreenPos();
    // Scale the current font's existing glyph geometry, keeping ReShade's font.
    ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), ImGui::GetFontSize() * scale,
        position, ink(color), title);
    const auto size = ImGui::CalcTextSize(title);
    ImGui::Dummy(ImVec2(size.x * scale, size.y * scale));
}
inline void help(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, rgb(0xadb8c8));
    ImGui::TextWrapped("%s", text);
    ImGui::PopStyleColor();
}
// Auto-height children provide native rounded backgrounds and clipping. Their
// heights are measured by ImGui, so expanded sections grow without fixed gaps.
struct PanelCard {
    bool visible;
    PanelCard(const char* id, const char* title, const char* description,
              unsigned accent = blue, unsigned background = 0x20242b) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, rgb(background));
        ImGui::PushStyleColor(ImGuiCol_Border, rgb(accent,.35f));
        visible = ImGui::BeginChild(id, ImVec2(0,0),
            ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding,
            ImGuiWindowFlags_HorizontalScrollbar);
        ImGui::PushItemWidth(std::max(90.f,ImGui::GetContentRegionAvail().x * .46f));
        if (visible && title && *title) {
            const auto position = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddRectFilled(position,
                ImVec2(position.x + 3, position.y + ImGui::GetTextLineHeight()), ink(accent), 1.5f);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 10);
            heading(title, accent);
            if (description && *description) help(description);
            ImGui::Spacing();
        }
    }
    ~PanelCard() { ImGui::PopItemWidth(); ImGui::EndChild(); ImGui::PopStyleColor(2); ImGui::Spacing(); }
    explicit operator bool() const { return visible; }
    PanelCard(const PanelCard&) = delete;
};
template <class Body> bool section_card(bool plain, const char* id, const char* title,
                                      const char* description, unsigned accent, Body body) {
    if (plain) return body();
    PanelCard card(id, title, description, accent);
    return card ? body() : false;
}
inline void fact_tile(const char* label, const char* value, float width, unsigned accent,
                      const char* tooltip) {
    const auto position = ImGui::GetCursorScreenPos();
    const float height = ImGui::GetTextLineHeight() * 3.6f;
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(position, ImVec2(position.x + width, position.y + height), ink(0x242c37), 6);
    draw->AddRect(position, ImVec2(position.x + width, position.y + height), ink(0x3a4655), 6);
    draw->AddRectFilled(ImVec2(position.x,position.y + 8), ImVec2(position.x + 3,position.y + height - 8), ink(accent), 1);
    const ImVec4 clip(position.x+7,position.y,position.x+width-6,position.y+height);
    draw->AddText(ImGui::GetFont(), ImGui::GetFontSize() * 1.2f,
        ImVec2(position.x + 11, position.y + 7), ink(0xecf0f5), value,nullptr,0,&clip);
    draw->AddText(ImGui::GetFont(), ImGui::GetFontSize() * .82f,
        ImVec2(position.x + 11, position.y + height - ImGui::GetFontSize() - 5), ink(0xadb8c8), label,nullptr,0,&clip);
    ImGui::Dummy(ImVec2(width,height));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s",tooltip);
}
} // namespace phoenix_ui
