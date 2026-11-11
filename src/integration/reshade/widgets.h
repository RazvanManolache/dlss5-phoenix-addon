#pragma once
#include <imgui.h>
#include <algorithm>
#include <string_view>
#include <string>

namespace phoenix_ui {
inline const char* slider_help(std::string_view label) {
    if(label=="Resolution scale")return "100% processes the captured source at its original size. 200% uses four times the pixels; 400% uses sixteen times. Higher scales enlarge the captured image rather than making the game render more geometry. They may change generated detail, cost substantial time/VRAM and are not guaranteed to improve quality. Backend texture limits still apply.";
    if (label == "Depth strength") return "Adjusts the contrast of the depth map in 0.001x steps. 0 disables depth, 1 preserves contrast, and higher values increase separation.";
    if (label == "Near depth") return "Near endpoint as a percentage of the observed depth span. Raise it to devote more contrast to the farther part of the scene. Values below it are clamped.";
    if (label == "Far depth") return "Far endpoint as a percentage of the observed depth span. Lower it to devote more contrast to the closer part of the scene. Values above it are clamped.";
    if(label=="NIS sharpness")return "Controls NVIDIA Image Scaling sharpening during enlargement. Start at 0.25; higher values can emphasize fine edges but create halos or exaggerated texture. It is independent of neural tone/structure and applies only when NIS actually runs. At 100% no resize is needed.";
    if(label=="Diffuse white (nits)")return "Sets the reference white used by the color conversion when Override diffuse white is enabled. Prefer automatic white unless you know the source's HDR reference. This is not a screen brightness control.";
    if(label=="Exposure multiplier")return "Changes the exposure of the signal presented to the model and its inverse conversion. Start at 1. Large changes can alter the model's interpretation of lighting and materials; this is not simply a final brightness adjustment.";
    if(label=="CPU threads"||label=="Portable CPU threads")return "Controls CPU inference parallelism. Zero restores automatic thread selection. More threads are not always faster; the useful count depends on the CPU and competing game work.";
    if(label.starts_with("Maximum result age"))return "Limits how old an enhanced frame may be when displayed. Larger values tolerate slower processing but can show more delayed motion. Very strict limits can discard completed results.";
    if(label=="Split position")return "Moves the boundary between comparison sources. This changes the comparison display, not the model or its quality.";
    if(label=="Difference gain")return "Amplifies the difference view so small pixel changes are easier to inspect. It does not strengthen the actual enhanced output.";
    if(label=="Overall intensity"||label=="Intensity")return "Controls the overall appearance blend. Zero retains the original appearance and 1 reaches full strength. The display blend saturates at 1, so values above 1 do not produce a stronger makeover.";
    if(label=="Local tone")return "Controls the model's local shading and tone treatment. Start at 1; higher values can produce a stronger makeover but may exaggerate lighting or material changes. Compare on both skin and the environment.";
    if(label=="Local structure")return "Controls the model's structural/detail treatment. Higher values can enhance perceived detail but may exaggerate edges or invent texture. More is not automatically more realistic.";
    if(label=="Skin structure")return "Controls structural treatment in the automatic skin mask. -1 follows Local structure. Higher values can strengthen facial detail but may exaggerate pores or alter skin texture. Requires the automatic skin mask.";
    if(label=="Motion X scale"||label=="Motion Y scale")return "Multiplies the engine motion-vector scale for this axis. Keep 1 unless the integration's motion convention is known to be incorrect. This is a guide correction, not a quality strength.";
    if(label=="Host exposure"||label=="Host exposure scale")return "Adjusts the legacy host exposure convention used around inference. Keep 1 for a neutral starting point. The Custom color and scaling path uses its own exposure controls instead.";
    if(label=="Host global tone")return "Adjusts the legacy host global-tone value supplied to the neural-rendering bridge. Start at 1; change it only while comparing the effect on lighting and materials.";
    if(label=="Reset frames after switch")return "Resets neural history for this many frames after changing the output or model. More resets can hide stale history but also delay stable temporal reuse.";
    if(label=="Reset every N frames (0 = off)")return "Periodically discards neural history. Zero keeps normal history behavior. Frequent resets can increase variation; use them to diagnose stale history rather than as a general quality improvement.";
    if(label=="Network passes per frame")return "Runs the network repeatedly on the preceding result. One pass is the trained starting point. Extra passes cost time and can exaggerate details or instability; they are not guaranteed quality improvements.";
    if(label=="Transformer blocks"||label=="Minimum blocks"||label.starts_with("Block "))return "Controls the learned network schedule. The full eight blocks with one repetition each are the trained starting point. Skipping or repeating blocks is an untrained approximation that can reduce realism and stability.";
    if(label=="Phoenix DLSS5 GPU budget (ms)")return "Target used by adaptive quality control. A lower target can reduce network work and quality. It is feedback, not a hard deadline that interrupts inference. Use the full fixed network first when judging maximum quality.";
    if(label=="Desired FPS")return "Game presentation-rate target for adaptive quality control. It does not impose a frame cap or guarantee that the game can reach this rate. CPU and other rendering work may remain the bottleneck.";
    if(label=="Capture burst frames")return "Number of diagnostic frames to capture. Zero disables the burst. Capturing can use substantial disk space and affect measured processing time.";
    if(label=="Motion analysis height"||label=="Minimum motion analysis height")return "Resolution used for estimated motion analysis, separate from neural output resolution. Higher values may capture finer motion but cost more. Lower values reduce motion-analysis work; start with the default and inspect moving edges.";
    if(label=="Motion consistency tolerance (pixels)")return "Rejects estimated motion that disagrees with reverse tracking beyond this tolerance. Smaller values reject more vectors; larger values trust more uncertain motion. Inspect both stability and ghosting when changing it.";
    if(label=="Motion cost threshold (per mille)")return "Rejects estimated motion above this normalized matching-cost threshold. A lower value is more selective and can cause more spatial fallbacks. A higher value can admit unreliable motion.";
    if(label=="GPU monitoring sample interval")return "Samples detailed GPU stages once every this many frames. Larger intervals reduce monitoring overhead but produce fewer timing samples. This does not change the trained model.";
    if(label=="Pipeline age budget (ms)")return "Latency target for adaptive pipeline controls. It can trade resolution or network work for responsiveness when those adjustments are enabled. It is not a guaranteed end-to-end latency.";
    if(label=="Minimum neural scale")return "Lowest working scale adaptive processing may select. Higher values preserve more detail; lower values allow larger speed/quality tradeoffs. 100% is the full-resolution quality reference.";
    if(label=="Minimum network passes")return "Lowest number of passes adaptive processing may use. One pass is the recommended trained starting point; extra repeated passes add cost and can exaggerate the image.";
    if(label=="Scaling sharpness")return "Sharpness used by the selected additional scaler. Higher values can make edges look crisper but may create halos. Inspect hair, text and bright boundaries rather than maximizing it.";
    if(label=="Chain brightness")return "Brightness treatment in the processing chain. Start at 0. It affects the image only when the chain's Color treatment stage is enabled, and stage order determines whether it occurs before or after inference.";
    if(label=="Chain contrast")return "Contrast treatment in the processing chain. Start at 1. Higher values can deepen shadows and clip bright detail; judge realistic lighting as well as immediate punch.";
    if(label=="Chain saturation")return "Color saturation treatment in the chain. Start at 1. A slight adjustment may suit a scene, but high saturation is not the same as more realistic materials.";
    if(label=="Chain gamma")return "Gamma treatment in the chain. Start at 1. Changes midtone brightness and can affect how the model interprets the scene when placed before inference.";
    if(label=="Fusion layers"||label=="Second layer scale"||label=="Third layer scale")return "Controls experimental multiscale fusion. More layers add work, and smaller layers see less fine detail. Fusion is an approximation; compare against the single full-resolution result before choosing it for realism.";
    if(label=="Comparison zoom")return "Magnifies the comparison around the selected focus point. This changes inspection, not neural processing resolution or model quality.";
    if(label=="Comparison focus X"||label=="Comparison focus Y")return "Moves the focus of the comparison zoom or loupe. The center is 0.5. This does not change model inference.";
    return "Adjust this value within the displayed range. Use R to restore this control's default without resetting other settings.";
}
inline const char* button_help(std::string_view id) {
    const auto label=id.substr(0,id.find("##"));
    if(id=="Defaults##processing")return "Restore Color and scaling to its defaults, including the normal canvas and one network pass. Use an individual R button to preserve the rest of your tuning.";
    if(label=="R")return "Restore this control's default value without resetting the other settings.";
    if(label=="Save configuration")return "Save the current Phoenix settings for this game. The saved configuration is used next time the game starts.";
    if(label=="Retry rendering"||label=="Retry preferred routes"||label=="Recheck runtimes / retry routes")return "Retry initialization and the preferred processing routes after a failure. This can restart processing; use it after fixing a missing model or runtime, rather than for ordinary tuning.";
    if(label=="Cancel")return "Cancel the pending action and keep the current configuration.";
    if(label=="Clear timings"||label=="Clear GPU timing")return "Clear retained timing measurements and charts. This does not change the model or image settings.";
    if(label=="Defaults")return "Restore the defaults for this section. Other sections retain their settings. Use the R beside a slider to reset only that value.";
    if(label=="Browse..."||label=="Browse")return "Choose the model or runtime location for this field. The selected files must match the model and processing route.";
    if(label=="Validate model")return "Check the selected model source and its required files. Successful validation does not certify a game's rendered image or compatibility.";
    if(label=="Reset neural controls")return "Restore neural controls to their default values. This changes several settings; use individual R buttons for smaller adjustments.";
    if(label=="Reset neural overrides"||label=="Restore game controls")return "Clear Phoenix's neural overrides and restore the game's own control values. This resets several overrides together; it does not replace the model files.";
    if(label=="Exit comparison")return "Return to the processed output without the comparison split. The selected model and tuning are kept.";
    if(label=="Swap A / B")return "Exchange the two comparison sources while keeping the same split position and comparison mode.";
    if(label=="Load preset")return "Apply the selected saved preset. Resource choices in the preset may require processing to be reconfigured.";
    if(label=="Save preset")return "Save the current controls as a reusable named preset. This is separate from saving the game's startup configuration.";
    if(label=="Use native NR")return "Select the NVIDIA neural-rendering output using the separately supplied compatible model DLL.";
    if(label=="Reset history")return "Discard the previous neural frame history. Useful after a scene change or visible ghosting; it does not reload model weights.";
    if(label=="Capture input burst")return "Capture the configured number of diagnostic input frames. Captures use disk space and can interrupt normal processing timing.";
    if(label=="Reset repetitions to defaults"||label=="Restore full network"||label=="Use full network for native x86 driver")return "Restore the trained full-network schedule. Skipping or repeating learned blocks is an untrained approximation and may reduce realism or stability.";
    if(label=="Use fixed blocks for portable processing")return "Disable unsupported adaptive block changes and use the selected fixed network schedule for this portable route.";
    if(label=="Clear")return "Remove this hotkey assignment. The action remains available in the panel.";
    if(label=="Reload overrides now")return "Read the external override configuration again and apply its supported values. Resource changes can reconfigure processing.";
    if(label=="Log marker")return "Write a marker into the diagnostic log to identify this point in the session. This does not change the image.";
    if(label=="Move up")return "Move this processing stage one position earlier. Stage order changes the image and may reconfigure the processing chain.";
    if(label=="Reset motion and pipeline options")return "Restore the motion, latency, processing-chain and inspection defaults. This resets several controls at once.";
    if(label=="Models and runner setup")return "Open Setup to choose model files, processing devices and available runtimes.";
    if(label=="More image controls")return "Open Image for resolution, color, reconstruction, motion and temporal-history settings.";
    if(label=="Compare input / processed")return "Show the original input beside the selected processed output using the comparison view.";
    if(label=="Save original + modified")return "Save an original/processed image pair from the same completed frame. Later game effects and overlay elements may be outside this capture point.";
    if(label=="Timing comparisons")return "Open Performance to compare retained timing and FPS measurements for each processing configuration.";
    if(label=="Use normal resolution for portable processing")return "Return to the full-resolution canvas required by this portable route. Other image settings are retained.";
    return "Activate this action. The processing status area reports its result.";
}
inline void item_help(const char* text) {
    if (!text || !ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) return;
    ImGui::BeginTooltip();ImGui::PushTextWrapPos(ImGui::GetFontSize()*28.f);
    ImGui::TextUnformatted(text);ImGui::PopTextWrapPos();ImGui::EndTooltip();
}
inline bool button(const char* label, const ImVec2& size=ImVec2(0,0), const char* help=nullptr) {
    const bool changed=ImGui::Button(label,size);item_help(help?help:button_help(label));return changed;
}
inline bool small_button(const char* label, const char* help=nullptr) {
    const bool changed=ImGui::SmallButton(label);item_help(help?help:button_help(label));return changed;
}
inline void reserve_reset_space(const char* label) {
    const float available=ImGui::GetContentRegionAvail().x;
    const float label_width=ImGui::CalcTextSize(label,nullptr,true).x;
    ImGui::SetNextItemWidth(std::max(64.f,std::min(ImGui::CalcItemWidth(),
        available-label_width-28.f-3*ImGui::GetStyle().ItemInnerSpacing.x)));
}
template<class T> inline bool slider_reset(const char* label,T* value,T initial) {
    const float right=ImGui::GetCursorScreenPos().x+ImGui::GetContentRegionAvail().x;
    if(ImGui::GetItemRectMax().x+28.f+ImGui::GetStyle().ItemSpacing.x<=right)ImGui::SameLine();
    ImGui::PushID(label);
    const auto id=std::string("R##default.")+label;
    const bool reset=button(id.c_str(),ImVec2(28,0));
    if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Reset %s to its default: %.4g",label,double(initial));
    if(reset)*value=initial;
    ImGui::PopID();return reset;
}
inline bool slider_float(const char* label,float* value,float minimum,float maximum,float initial,
    const char* format="%.3f",ImGuiSliderFlags flags=0) {
    reserve_reset_space(label);
    const bool changed=ImGui::SliderFloat(label,value,minimum,maximum,format,flags);
    if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::BeginTooltip();ImGui::PushTextWrapPos(ImGui::GetFontSize()*28.f);
        ImGui::TextUnformatted(label);ImGui::Separator();ImGui::TextUnformatted(slider_help(label));
        ImGui::Separator();ImGui::Text("Range: %.4g to %.4g. Default: %.4g.",double(minimum),double(maximum),double(initial));
        ImGui::TextUnformatted("Ctrl+click to type a value; R restores the default.");
        ImGui::PopTextWrapPos();ImGui::EndTooltip();
    }
    return slider_reset(label,value,initial)||changed;
}
inline bool slider_int(const char* label,int* value,int minimum,int maximum,int initial,
    const char* format="%d",ImGuiSliderFlags flags=0) {
    reserve_reset_space(label);
    const bool changed=ImGui::SliderInt(label,value,minimum,maximum,format,flags);
    if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::BeginTooltip();ImGui::PushTextWrapPos(ImGui::GetFontSize()*28.f);
        ImGui::TextUnformatted(label);ImGui::Separator();ImGui::TextUnformatted(slider_help(label));
        ImGui::Separator();ImGui::Text("Range: %d to %d. Default: %d.",minimum,maximum,initial);
        ImGui::TextUnformatted("Ctrl+click to type a value; R restores the default.");
        ImGui::PopTextWrapPos();ImGui::EndTooltip();
    }
    return slider_reset(label,value,initial)||changed;
}
} // namespace phoenix_ui
