# Phoenix DLSS5 1.3.0

## Better support for older games

- **Keep capture, inference, and display moving.** Bounded queues overlap work across frames on supported routes. DX9 skips busy readbacks and keeps processed output paired with the frame it came from. CPU-staged helper routes reuse transfer buffers and can prepare the next input while inference runs.
- **Choose a motion estimator.** Automatic prefers engine vectors, then FidelityFX, then NVIDIA Optical Flow when available. The Windows addon also offers OpenCV DIS, D3D12 Video Motion Estimation, and a built-in luma block matcher. Choose Zero motion for spatial processing with history reset. [Motion providers and requirements](motion-providers.md)
- **Keep DX9 depth current.** Phoenix tracks depth freshness independently of color and checks it around focus and resource changes. Compatible game depth can guide Phoenix and NVIDIA DLSS5 or be inspected live using **Depth only**. [Depth controls](USER_GUIDE.md#legacy-depth)

## Routing and diagnostics

- **Recover more reliably on Windows.** The NVIDIA helper can retry feature warm-up without publishing the input as a processed frame. Matching WOW64 packets are checked before startup. Options that need an external x64 helper on x86 explain the switch in their tooltips and restore the saved execution preference when deselected.
- **See frame latency and game architecture.** The overlay identifies the graphics API and x86/x64 process. The round-trip card measures capture through composition where frame timing can be matched. Game FPS, fresh processed FPS, and neural timing remain separate.

Start with [DLSS5 Everything](https://github.com/RazvanManolache/DLSS5-Everything) or follow the [manual installation guide](INSTALLATION.md). Packages keep model files separate. [Validation and compatibility limits](VALIDATION_1.3.0.md)
