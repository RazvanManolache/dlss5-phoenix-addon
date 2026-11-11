# Phoenix DLSS5 1.2.0

## Game depth and scaling

- **Game depth** feeds compatible DirectX depth attachments to Phoenix and NVIDIA DLSS5 on the Windows readback helper. DX9 can resolve compatible multisampled depth on NVIDIA hardware while retaining the game's antialiasing. Capture tracks depth-writing passes and rejects stale samples when the view changes.
- **Depth only** displays the live depth map. Strength, inversion and range controls tune the preview and the depth guide. Auto prefers usable game depth, then tries the configured estimators. Game depth, NVIDIA, Small and None are available as explicit choices. The optional estimators require a separate provider and model files. [Depth setup and controls](USER_GUIDE.md#legacy-depth).
- **Color and scaling** supports working resolutions up to 400% per dimension, with selectable filters including NVIDIA Image Scaling on supported routes. Tooltips describe the choices and their tradeoffs. [Color and scaling guide](USER_GUIDE.md) · [Example tuning](REALISM_TUNING.md).

## Controls and responsiveness

- Live tuning applies compatible edits without reloading the model. The CPU-transfer helper can retain its model across focus-related runtime resets and discard the pre-reset output on resume.
- The overlay reports game FPS separately from processed FPS. Timing history records how often processing produces a new frame alongside the game's presentation rate.
- Sliders have reset controls, and control tooltips explain what each setting changes. Hook choices reflect the route and inputs available in the current game.

## Packages and validation

This release includes Windows x86/x64 add-ons, the RTX Compatibility variant, portable runtimes, the optional input companion, Linux Vulkan-layer and native SDK packages, and the source archive. Model weights are separate.

[Validation details](VALIDATION_1.2.0.md).
