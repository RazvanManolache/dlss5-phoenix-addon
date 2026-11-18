# Phoenix DLSS5 changelog

The [README](README.md) shows the latest dated release. Earlier highlights are
kept here; each linked release note includes its validation details.

## 1.3.1 — 2026-11-16

- **Keep automation attached to the game.** The optional Phoenix Input companion can bind to a specific game executable in an install shared with its launcher.
- **Follow viewport changes.** Phoenix Input follows the active game window and preserves each window’s original message handler.

[1.3.1 release notes](docs/RELEASE_1.3.1.md).

## 1.3.0 — 2026-11-14

- **Faster frame delivery for older games.** Capture, helper processing, and display can overlap on supported routes. DX9 avoids waiting on busy captures and keeps each processed result paired with its source frame.
- **Motion for games without engine vectors.** Choose FidelityFX, NVIDIA Optical Flow, OpenCV DIS, D3D12 Video Motion Estimation, or a built-in luma block matcher.
- **More dependable game depth.** DX9 tracks fresh depth independently of color, helping keep the depth preview and guide current through camera movement and focus changes.
- **Better Windows compatibility and diagnostics.** Matched WOW64 packages, improved helper startup, clearer x64 fallback, and a pipeline round-trip display next to game FPS and processed FPS.

[1.3.0 release notes](docs/RELEASE_1.3.0.md).

## 1.2.0 — 2026-11-11

- **Game depth for older DirectX games:** use compatible engine depth with Phoenix or NVIDIA DLSS5, including NVIDIA DX9 multisampled depth. Inspect it live in **Depth only**, or configure an optional estimator for games without usable depth.
- **More control over scaling:** choose a resize filter, including NVIDIA Image Scaling on supported routes, and work at up to 400% resolution per dimension.
- **Clearer controls and performance:** contextual tooltips, slider resets, and separate game and processed FPS readings.
- **Less interruption while tuning:** live edits reuse the loaded model, and compatible helpers resume after focus-related resets.

[1.2.0 release notes](docs/RELEASE_1.2.0.md).

## 1.1.6 — 2026-11-10

- Fixed appearance with super resolution and custom processing scales.
- Kept resolution choices usable with custom color controls and displayed the active working scale in **Color and scaling**.

[1.1.6 fixes and validation](docs/RELEASE_1.1.6.md).

## 1.1.5

- **Consistent appearance:** shared style and intensity controls for Phoenix and NVIDIA, with corrected blending above intensity 1.
- **One-click choices:** equal-sized buttons for short option lists, simpler Image controls and a separate Hotkeys tab.
- **Quick bypass:** rebindable F6 toggles processing while preserving your selected model and settings.
- **Useful FPS comparisons:** game FPS in the main panel, history chart and retained timing grid.
- **Reliable controls:** temporary response-file locks retry without repeating commands or reporting a rendering failure.

[1.1.5 fixes and validation](docs/RELEASE_1.1.5.md).

## 1.1.0

- **Faster native CUDA:** close to native RTX and NVIDIA speed on the tested RTX 5080, for Original and Lightweight FP8.
- **Linux support:** a per-game Vulkan layer, portable CPU/CUDA/Vulkan processing and an optional native CUDA executor.
- **Temporal and extended Windows functionality:** persistent history, motion and guide inputs, masks, appearance controls and shared D3D12/CUDA texture transport on supported routes.
- **Faster processing across backends:** reduced host overhead and improved portable CUDA, Vulkan and CPU execution.

**2,219 validation checks passed; tested across 76 games.**
[Release notes and measurements](docs/RELEASE_1.1.0.md) · [Validation details](docs/VALIDATION_1.1.0.md).

Older releases are documented in the [release archive](https://github.com/RazvanManolache/dlss5-phoenix-addon/releases).
