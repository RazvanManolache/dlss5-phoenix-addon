# Phoenix DLSS5 changelog

The [README](README.md) shows the latest dated release. Earlier highlights are
kept here; each linked release note includes its validation details.

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
