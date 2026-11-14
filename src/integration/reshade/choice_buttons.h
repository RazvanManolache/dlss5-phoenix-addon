#pragma once
#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include "choice_help.h"

namespace phoenix_ui {
// Small, fixed option sets take one click. Every option in a group has the
// same hit area; narrow panels wrap rows without shrinking the targets.
inline bool choice_buttons(const char* label,int* value,const char* const* items,int count,
                           const char* const* captions=nullptr,const uint8_t* enabled=nullptr,
                           const char* const* route_notes=nullptr) {
    if(count<=0)return false;
    ImGui::TextUnformatted(label);
    const auto& style=ImGui::GetStyle();
    const float gap=style.ItemSpacing.x, available=std::max(1.f,ImGui::GetContentRegionAvail().x);
    float minimum=96.f;
    std::vector<std::string> display;
    for(int i=0;i<count;++i) {
        std::string text=captions?captions[i]:items[i];
        if(!captions)if(const auto parenthesis=text.find(" (");parenthesis!=std::string::npos)text.resize(parenthesis);
        minimum=std::max(minimum,ImGui::CalcTextSize(text.c_str()).x+2*style.FramePadding.x+8);
        display.push_back(std::move(text));
    }
    const int maximum=std::clamp(int((available+gap)/(minimum+gap)),1,count);
    const int rows=(count+maximum-1)/maximum, columns=(count+rows-1)/rows;
    const float width=std::max(1.f,std::floor((available-gap*(columns-1))/columns));
    const float height=std::max(28.f,ImGui::GetFrameHeight()*1.25f);
    bool changed=false;
    ImGui::PushID(label);
    for(int i=0;i<count;++i) {
        if(i%columns)ImGui::SameLine(0,gap);
        ImGui::BeginDisabled(enabled && !enabled[i]);
        const bool selected=*value==i;
        ImGui::PushStyleColor(ImGuiCol_Button,style.Colors[selected?ImGuiCol_HeaderActive:ImGuiCol_Button]);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize,selected?1.f:0.f);
        const auto id=display[i]+"###choice."+label+"|"+items[i];
        if(ImGui::Button(id.c_str(),ImVec2(width,height)) && !selected) {*value=i;changed=true;}
        if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize()*28.f);
            ImGui::TextUnformatted(items[i]);
            if (const char* help=choice_help(label,i)) {
                ImGui::Separator();
                ImGui::TextUnformatted(help);
            }
            if (route_notes && route_notes[i]) {
                ImGui::Separator();
                ImGui::TextUnformatted(route_notes[i]);
            }
            if (enabled&&!enabled[i]) {
                ImGui::Separator();
                ImGui::TextUnformatted("Unavailable on the current route.");
            }
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
        ImGui::PopStyleVar();ImGui::PopStyleColor();ImGui::EndDisabled();
    }
    ImGui::PopID();
    return changed;
}
}
