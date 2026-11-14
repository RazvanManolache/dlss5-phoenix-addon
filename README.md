# Phoenix DLSS5

My attempt at an open-source runner for NVIDIA DLSS5 neural rendering, packaged as a ReShade addon. Bring a compatible model, give a favourite game another look, and decide for yourself whether you like what it does.

I dislike how much of the software we add to our games is closed source. We rush for the best experience without being able to inspect what we're running. I'd rather let you look under the hood. Better pixels shouldn't require blind trust.

I know DLSS5 hasn't exactly won a popularity contest. I'm curious about what it can do for the games we already love—and, eventually, looks less interested in counting pores. [Why I'm making this](docs/PROJECT.md#why-im-making-this).

[Download 1.3.0](https://github.com/RazvanManolache/dlss5-phoenix-addon/releases/tag/1.3.0) ·
[Installation](docs/INSTALLATION.md) · [User guide](docs/USER_GUIDE.md) · [All documentation](docs/README.md)

## New in 1.3.0 — 2026-11-14

- **Faster frame delivery for older games.** Capture, helper processing, and display can overlap on supported routes. DX9 avoids waiting on busy captures and keeps each processed result paired with its source frame.
- **Motion for games without engine vectors.** Choose FidelityFX, NVIDIA Optical Flow, OpenCV DIS, D3D12 Video Motion Estimation, or a built-in luma block matcher.
- **More dependable game depth.** DX9 tracks fresh depth independently of color, helping keep the depth preview and guide current through camera movement and focus changes.
- **Better Windows compatibility and diagnostics.** Matched WOW64 packages, improved helper startup, clearer x64 fallback, and a pipeline round-trip display next to game FPS and processed FPS.

[1.3.0 release notes](docs/RELEASE_1.3.0.md) · [Earlier changes](CHANGELOG.md).

<table><tr><td><a href="docs/features/overlay-play.png"><img src="docs/features/overlay-play.png" alt="Phoenix Play tab with model choices and separate game and processed FPS" width="420"></a></td><td><a href="docs/features/overlay-performance.png"><img src="docs/features/overlay-performance.png" alt="Phoenix timing and FPS comparisons" width="420"></a></td></tr></table>

*Current C++/ImGui controls rendered with synthetic status values. [More UI screenshots](docs/USER_GUIDE.md#a-quick-look).*

## The easiest way to try Phoenix

DLSS5 Everything is a free, open-source Windows app that gets Phoenix running in your games. It finds installed games, detects their graphics API and whether they are 32-bit or 64-bit, and installs the matching Phoenix add-on and runtime components. You can verify or repair an installation, launch the game from the library, and restore backed-up files if you want to undo the setup.

It also gives you per-game model and effect settings and can use Phoenix on windows, the desktop or a webcam. Open the app, pick a game, choose **Install**, and then **Play**.

[Download DLSS5 Everything](https://github.com/RazvanManolache/DLSS5-Everything/releases/latest) ·
[Source code and setup guide](https://github.com/RazvanManolache/DLSS5-Everything)

<!-- current-two-model-benchmark -->
## Native RTX: the main event

**Performance on par with NVIDIA DLSS5 at 1080p on the tested RTX 5080.** Both Original and Lightweight have comparable isolated inference latency; this is not end-to-end game FPS or portable-backend performance.

| Model at 1080p | Phoenix DLSS5 GPU ms | NVIDIA DLSS5 GPU ms | Phoenix DLSS5 VRAM MiB | NVIDIA DLSS5 VRAM MiB |
| --- | ---: | ---: | ---: | ---: |
| Original | 3.841 | 3.848 | 574.6 | 639.6 |
| Lightweight FP8 | 2.250 | 2.248 | 540.8 | 574.9 |

**RAM gets a seat at the table too.** Resident RAM (Phoenix DLSS5 / NVIDIA DLSS5): **171.3 / 333.6 MiB for Original**, **170.2 / 300.9 MiB for Lightweight**. VRAM is GPU-local residency; both are process snapshots including harness/driver overhead, not whole-game totals or peak requirements. [Memory details](docs/benchmarks/current/README.md#memory-at-1080p).

Measured 2026-10-30; full graphs, one pass. Small margins vary between runs; higher-resolution differences, including Original at 4K, are covered in the full report.

[Performance guide](docs/PERFORMANCE.md) · [Measurements and methodology](docs/benchmarks/current/README.md) · [1080p scaling investigation](docs/NATIVE_SCALING_PROFILE.md)
<!-- /current-two-model-benchmark -->

## See it in games

[See the matching original and processed captures from seven games, including Batman: Arkham Knight, Hitman: Absolution and DOOM II](docs/GAME_GALLERY.md).

## Features

- **Pick the look:** Phoenix DLSS5, NVIDIA DLSS5, split views and difference views.
- **Two model sizes:** Original and Lightweight FP8, plus compatible external model sources and a shared model library.
- **Tune live:** tone, structure, skin, masks, intensity and appearance styles.
- **32-bit and 64-bit games:** automatic execution choices with an x64 helper when needed.
- **Stay in the game process when possible:** local D3D11/D3D12 bridges and experimental x86 library routes retain checked helper fallback.
- **Overlap capture and inference:** bounded frame queues keep older games responsive and preserve matching input/output frames.
- **Recover motion and depth in older games:** estimate missing motion, use compatible game depth when available, and preview the live depth guide.
- **Keep up with driver changes:** experimental x86 bridges discover and validate the installed NVIDIA driver, with results cached per driver and runtime package.
- **Broad capture integration:** D3D9/10/11/12, OpenGL and Vulkan, with route-specific requirements.
- **Build a processing chain:** scaling, color, optional multiscale fusion and motion estimation where supported.
- **See where time goes:** separate game and processed FPS, neural timing, capture-to-presentation latency, and matching original/processed screenshots.
- **Use it beyond ReShade:** a native SDK and file/framed-stream interfaces.

[Full feature catalog](docs/FEATURES.md) · [Controls and screenshots](docs/USER_GUIDE.md)

Older RTX kernels are experimental. Hardware testing here uses an RTX 5080. [Compatibility and tested hardware](docs/COMPATIBILITY.md).

Windows motion sources include NVIDIA Optical Flow, AMD FidelityFX, OpenCV DIS, and D3D12 Video Motion Estimation on compatible routes. [Provider requirements and benchmarks](docs/motion-providers.md).

## Manual installation

The native Linux preview provides a per-game Vulkan layer and optional CUDA backends. [Linux guide](docs/LINUX.md) · [Native CUDA guide](docs/LINUX_NATIVE_CUDA.md).

1. [Choose and install the package](docs/INSTALLATION.md) with full-addon-support ReShade.
2. Select a separately supplied compatible model in Setup.
3. Start with Automatic execution, compare the result, then [tune the look](docs/USER_GUIDE.md).

## Read more

| What you need | Guide |
| --- | --- |
| Installation, package choice and updates | [Installation](docs/INSTALLATION.md) |
| Game screenshots and processing routes | [Gameplay comparison gallery](docs/GAME_GALLERY.md) |
| Features, controls and overlay screenshots | [Feature catalog](docs/FEATURES.md) · [User guide](docs/USER_GUIDE.md) |
| x86, GPU generations, graphics APIs and execution routes | [Compatibility](docs/COMPATIBILITY.md) · [Support matrix](docs/SUPPORT.md) |
| Performance, memory, scaling and historical studies | [Performance guide](docs/PERFORMANCE.md) |
| Build, package or integrate Phoenix DLSS5 | [Developer recipes](docs/DEVELOPMENT.md) · [Native SDK](docs/LIBRARY.md) |
| Motivation, contributions and upstream thanks | [About the project](docs/PROJECT.md) · [Credit ledger](docs/UPSTREAM_CREDITS.md) |

## License

Original project code is [Apache 2.0](LICENSE). Third-party components retain their own terms: [THIRD_PARTY.md](THIRD_PARTY.md) · [NOTICE](NOTICE). Separately supplied NVIDIA models and runtimes retain their own licenses.
