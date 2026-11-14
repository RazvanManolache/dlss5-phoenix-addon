#pragma once
#include <string_view>

namespace phoenix_ui {
// Recommendations describe the current processing equations and their tradeoffs.
// Encoding controls describe the source signal; they are not artistic presets.
inline const char* choice_help(std::string_view group, int option) {
    const auto pick = [option](const auto& tips) -> const char* {
        return option >= 0 && option < int(sizeof(tips) / sizeof(tips[0])) ? tips[option] : nullptr;
    };
    if (group == "Resolution mode") {
        static constexpr const char* tips[]{
            "Recommended for maximum image quality. Processes the full image, retaining the most source detail. A good starting point for a strong photorealistic makeover; appearance strengths still control the transformation.",
            "Processes a smaller image and reconstructs the full output. Faster, with less fine detail available to the model. Start with Quality if you need more speed. Color and appearance settings are preserved."};
        return pick(tips);
    }
    if (group == "Processing quality") {
        static constexpr const char* tips[]{
            "Best detail among the reduced-resolution presets: about 67% of each dimension. Recommended first compromise when Normal is too slow.",
            "Half of each dimension, or one quarter of the pixels. Saves more processing time but can soften faces, hair and text. Bicubic reconstruction may look sharper but cannot restore missing source detail.",
            "One third of each dimension, or about one ninth of the pixels. Largest quality sacrifice. Use for a strict performance limit rather than maximum realism."};
        return pick(tips);
    }
    if (group == "Scaling domain") {
        static constexpr const char* tips[]{
            "Filters in a signed nonlinear scRGB domain. Keeps negative values and compresses bright values during filtering. The established default; compare it with Linear around bright edges and skin.",
            "Filters in a signed perceptual PQ domain. Changes how dark and bright detail are weighted during resizing. Useful for comparison on wide-range images; it does not turn SDR footage into HDR.",
            "Filters linear-light BT.709 values. A useful starting point for faithful light averaging, especially with Area downsampling. Bright edges may look different from nonlinear filtering. No domain is universally best for every scene."};
        return pick(tips);
    }
    if(group=="Upscaling method") {
        static constexpr const char* tips[]{
            "Preserve the existing manual stage filters. The first prepares the neural canvas and the second restores the output. Existing profiles retain this mode and their image settings.",
            "Select by resize direction. Native GPU sRGB enlargement uses NVIDIA Image Scaling; reductions use Area. Other signals and CPU processing use Bicubic enlargement and Area reduction. Larger than 2x enlargement uses multiple NIS passes. At equal dimensions resizing is bypassed.",
            "Use NVIDIA's spatial Image Scaling shader for supported sRGB enlargement before or after the neural model, according to scale. Up to 400% uses staged NIS passes. Reduction uses the downsample filter. Requires a native GPU/helper route; this does not enable temporal DLSS Super Resolution or fabricate engine guides.",
            "Use Bilinear interpolation whenever enlargement is needed, either into a larger neural canvas or back to the output. Inexpensive and smooth, but can look softer. Reduction uses the separate downsample filter.",
            "Use Catmull-Rom Bicubic interpolation whenever enlargement is needed. Sharper than Bilinear, with possible edge halos/ringing. Reduction uses the separate downsample filter.",
            "Use Nearest-neighbor enlargement. Produces visible pixel blocks and is primarily useful for diagnostics or intentionally pixelated results. Reduction uses the separate downsample filter."};return pick(tips);
    }
    if (group == "Downsample filter") {
        static constexpr const char* tips[]{
            "Recommended when reducing resolution. Integrates the source footprint instead of picking a few samples, reducing shimmer and aliasing in hair, fabric and thin lines. It can look smoother than an aliased alternative.",
            "Interpolates nearby pixels. A simple, inexpensive option, but large reductions can miss fine detail and alias more than Area.",
            "Selects individual source pixels. Produces blockier edges and aliasing; mainly useful for inspection or deliberately pixelated results."};
        return pick(tips);
    }
    if (group == "Upsample filter") {
        static constexpr const char* tips[]{
            "Smooth, stable interpolation. Recommended if sharper reconstruction produces halos or ringing. It can look softer than Bicubic at a reduced working scale.",
            "Catmull-Rom bicubic interpolation. Recommended first for a sharper reduced-resolution result. Inspect hair, text and bright edges for halos or overshoot; switch to Bilinear if those appear.",
            "Repeats nearby samples without smoothing. Produces visible pixel blocks. Intended for inspection rather than a realistic reconstructed image."};
        return pick(tips);
    }
    if (group == "Scale reconstruction") {
        static constexpr const char* tips[]{
            "Adds the reconstructed neural change to the full-resolution original. Recommended for retaining original texture and fine detail when processing at a smaller scale. It also retains more of the original game's rendering style.",
            "Reconstructs the complete processed image. Try this for a stronger makeover with less of the original texture carried through. At reduced scales it can lose fine detail or soften text; compare with RGB residual."};
        return pick(tips);
    }
    if (group == "Input encoding") {
        static constexpr const char* tips[]{
            "Recommended. Uses the source information supplied by the game or capture route. Choose a manual encoding only when you know Auto is receiving incorrect metadata.",
            "Use only for a source whose RGB values are linear BT.709 light. Selecting this for an ordinary sRGB screenshot changes the interpretation of its brightness; it is not a quality upgrade.",
            "Use for an actual sRGB-encoded SDR source. Common for ordinary screenshots and byte-format final frames. Prefer Auto when its source metadata is correct.",
            "Use only for actual HDR10: PQ transfer with BT.2020 primaries. Selecting it for SDR causes incorrect brightness and color. This does not enable HDR capture or output on an SDR-only route.",
            "Use for an actual linear scRGB source, with 80 nits per unit. Typical of some Windows HDR paths. Match the source; this is not an artistic HDR effect.",
            "Use for a source already encoded in signed nonlinear scRGB. A specialized transfer interpretation, not a general recommendation for SDR games."};
        return pick(tips);
    }
    if (group == "NR signal encoding") {
        static constexpr const char* tips[]{
            "Recommended starting point for the current model bridge. Encodes the neural input as sRGB and decodes the result afterward. Keep this while comparing artistic strengths so signal changes do not confound the comparison.",
            "Uses an SDR-range PQ signal around inference. Changes the distribution of values seen by the model; it is not automatically more realistic or higher quality. Compare carefully for brightness and material changes.",
            "Uses a linear SDR signal around inference. Changes what the model sees even when the source is still decoded correctly. An experimental comparison choice; prefer sRGB until you have checked the result."};
        return pick(tips);
    }
    if (group == "Use exposure value") {
        static constexpr const char* tips[]{
            "Ignores the exposure multiplier and uses unity. Useful as a neutral reference while investigating brightness changes.",
            "Applies the multiplier to the exposure supplied to the processing bridge. Start at 1. This changes the signal the model sees; it is not simply a final-image brightness slider. Effective exposure is bounded to 0-10x.",
            "Uses the reciprocal of the effective exposure. For sources using the opposite exposure convention. Select only when that convention is known; it is not an extra quality mode."};
        return pick(tips);
    }
    if (group == "UI correction policy") {
        static constexpr const char* tips[]{
            "Recommended. Follows the host's UI policy and available coverage. A final-color-only capture cannot invent an engine UI mask.",
            "Disables the requested UI correction. The neural pass may affect HUD and text along with the scene.",
            "Requests UI correction using real engine-supplied UI or alpha coverage. Requires that coverage; it cannot work from color alone. This setting does not improve scene realism."};
        return pick(tips);
    }
    if (group == "Artistic style" || group == "Style") {
        static constexpr const char* tips[]{
            "Neutral artistic style. Recommended reference for judging the neural model's own makeover and changes from local tone, structure and skin strength.",
            "An additional natural-style treatment with a mild exposure, contrast and saturation adjustment. Compare against Default; its name does not guarantee a more realistic result in every game.",
            "A cinematic treatment that includes reduced saturation. Can suit stylized lighting, but is not a higher-quality model. For a strong makeover, compare this with Default before adding more grading."};
        return pick(tips);
    }
    if(group=="Output") {
        static constexpr const char* tips[]{"Show the unprocessed input. Keeps model and image settings for switching back.",
            "Run NVIDIA's compatible model DLL. Device support and the selected model DLL determine availability.",
            "Compare original, NVIDIA and Phoenix sources using the comparison controls. Evaluating both neural outputs adds work.",
            "Run the Phoenix implementation on the selected processing device. Use the same model and appearance controls when comparing with NVIDIA."};return pick(tips);
    }
    if(group=="Neural model")return "Choose the learned network architecture. Original and Lightweight FP8 use different encoder/decoder layouts and can produce different details. Lightweight usually costs less. Switching architecture loads its matching weights; it is separate from artistic style.";
    if(group=="NVIDIA neural model")return "Choose NVIDIA's architecture preset. Match Phoenix is recommended for a fair comparison. DLL default is version dependent; Lightweight FP8 requires a DLL containing that model. Changing this can recreate the NVIDIA feature.";
    if(group=="Model source")return "Select where model weights come from. Configured source uses Phoenix.json; a native model folder needs a compatible manifest and weights; DLL selection extracts a supported network. Model files remain your separate input.";
    if(group=="Native RTX kernel style")return "Select a compiled kernel implementation supported by this GPU. Automatic is recommended. This changes execution code, not artistic style or learned weights; a manual choice is useful for a measured compatibility or performance comparison.";
    if(group=="Processing aspect")return "Choose the shape of the neural working image. Keep the source aspect for natural proportions. Other ratios fit the processing canvas to that shape and can introduce borders; they do not change the game's camera.";
    if(group=="Upscaler")return "Choose the reconstruction provider after reduced-resolution processing. Spatial uses the selected filters. Temporal providers need a supported GPU, runtime and real frame guides; unavailable providers remain disabled.";
    if(group=="Hook method") {
        static constexpr const char* tips[]{
            "Disable processing at this insertion point. The saved color, scale and model choices are retained. Use the main processing toggle for a quick before/after comparison.",
            "Use an observed D3D12 render-resolution image before upscaling. Requires real engine feature buffers. This is unavailable on a final-frame capture route; choose Present there.",
            "Use the image from an observed D3D12 upscaling feature. Requires real engine feature buffers. This is unavailable on a final-frame capture route; choose Present there.",
            "Process the final frame. Recommended for compatibility capture routes, which cannot supply the earlier Render/Upscaled insertion points. Color and scaling controls still apply to the captured image."};
        return pick(tips);
    }
    if(group=="Present DLSS guide reuse") {
        static constexpr const char* tips[]{"Do not borrow guides from an observed DLSS feature. Final-color processing remains available without them.",
            "Recommended starting point. Reuse matching observed guides when available, otherwise continue with the supported fallback.",
            "Require matching observed guides before processing. Use for integration diagnostics; processing may wait if the game supplies no suitable feature."};return pick(tips);
    }
    if(group=="Depth")return "Choose the depth convention for real engine depth. Game depth follows the integration; the overrides supply a constant guide. Final-color capture has no geometry depth to adjust.";
    if(group=="Host HDR declare")return "Declare the signal convention used to create the NVIDIA feature. Match the actual host bridge; this is not an HDR enhancement effect. Changing a feature-create flag can reinitialize processing.";
    if(group=="Motion source") {
        static constexpr const char* tips[]{"Recommended. Prefer real engine vectors, then FidelityFX Optical Flow, then NVIDIA Optical Flow. Missing or rejected guides reset history.",
            "Use real engine-supplied vectors only. A final-color screenshot does not contain these guides; missing vectors fall back to spatial processing.",
            "Estimate motion with NVIDIA Optical Flow on supported hardware/routes. Estimation adds cost and can reject uncertain motion; inspect moving boundaries for ghosting.",
            "Use no motion guides. Processing resets recurrence when trustworthy motion is absent. Useful as a spatial reference; it may show more frame-to-frame variation.",
            "Estimate motion on the GPU with AMD FidelityFX shaders. Uses 8 x 8 blocks and works across GPU vendors on supported Direct3D 12 devices.",
            "Estimate dense motion with OpenCV DIS UltraFast on the CPU. Analysis reads back grayscale pixels and uploads the resulting vectors.",
            "Estimate motion on the Direct3D 12 video engine when the driver supports NV12 motion estimation at the selected analysis size.",
            "Estimate motion with a lightweight GPU shader. Matches 8 x 8 luminance blocks against the previous frame and rejects uncertain matches. Fast movement, fine edges and flat regions can lose temporal detail."};return pick(tips);
    }
    if(group=="Motion diagnostic view")return "Switch between the processed image and estimated-motion diagnostics. Direction/magnitude and confidence views help find tracking failures; they do not increase enhancement quality.";
    if (group == "Depth source") {
        const char* tips[]{
            "Prefer a verified game depth buffer. Otherwise try NVIDIA NvDepthAnythingV2 on the GPU, Depth Anything V2 Small on GPU or CPU, then continue without depth.",
            "Use game depth. DX9 requires a readable ReShade INTZ surface with depth writes in the current frame. DX10.1 and DX11 can copy a compatible attachment that matches the captured color target.",
            "Estimate relative depth with NVIDIA's NvDepthAnythingV2 Large model. Requires the optional ONNX depth provider and NVIDIA's deployable model.",
            "Estimate relative depth from each captured color frame using Depth Anything V2 Small. Requires the optional ONNX depth provider and Small model.",
            "Disable estimation and real-depth capture. Process without a depth guide."};
        return pick(tips);
    }
    if(group=="Pipeline monitoring")return "Off avoids optional monitoring; Basic collects CPU counters; Detailed samples GPU stages at the chosen interval. Detailed measurements add some overhead and do not represent total game frame time.";
    if(group=="Adaptive priority")return "Fixed preserves the chosen quality. Responsiveness or Image quality allow enabled adaptive controls to balance work against the age budget. Adaptive changes can affect detail; use Fixed when comparing maximum image quality.";
    if(group=="Additional upscaler")return "Choose one extra reconstruction provider for the processing chain. Existing uses the configured spatial reconstruction; NIS is a spatial scaler/sharpener; Tiny needs its installed 2x model and matching canvas. RTX VSR remains unavailable on unsupported routes.";
    if(group=="Target")return "Choose the signal used by adaptive block control: neural GPU time or game presentation FPS. FPS includes other game work, so reducing neural blocks cannot fix every bottleneck. Keep the full fixed network when judging realism.";
    if(group=="View")return "Choose how the comparison is displayed. Split, difference and diagnostic views change inspection, not model weights. Difference views amplify changes and are not the normal processed output.";
    if(group=="Source A"||group=="Source B")return "Select the input or neural output shown on this side of the comparison. Evaluating both Phoenix and NVIDIA can add processing cost. Keep model, style and strengths matched when comparing implementations.";
    if(group=="Preset")return "Select one of four saved setting slots. Selecting a slot does not apply it: Load preset restores it and Save preset replaces it with the current configuration.";
    if(group=="Log level")return "Choose diagnostic detail. Normal is the everyday starting point. More verbose logging is useful for troubleshooting and can add overhead or disk activity.";
    if(group=="Execution mechanism")return "Choose where inference runs. Automatic tries supported same-process routes before the isolated x64 helper. Explicit routes are for compatibility/testing; changing mechanism reinitializes processing.";
    if(group=="Helper transfer")return "Choose how frames reach the isolated helper. Automatic uses shared GPU textures where supported. CPU staging works on more APIs but adds capture, transfer and return latency. Shared textures currently require D3D11.";
    return "Select this option directly. The highlighted button is the current choice; disabled choices require capabilities unavailable on the active route.";
}
} // namespace phoenix_ui
