# Phoenix DLSS5

My attempt at an open-source runner for NVIDIA DLSS5 neural rendering,
packaged as a ReShade addon. Bring a compatible model, give a favourite game
another look, and decide for yourself whether you like what it does.

I dislike how much of the software we add to our games is closed source. We
rush for the best experience without being able to inspect what we're running.
I'd rather let you look under the hood. Better pixels shouldn't require blind trust.

I know DLSS5 hasn't exactly won a popularity contest. I'm curious about what
it can do for the games we already love—and, eventually, looks less interested
in counting pores. [Why I'm making this](docs/PROJECT.md#why-im-making-this).

[Download 1.0.1](https://github.com/RazvanManolache/dlss5-phoenix-addon/releases/tag/1.0.1) ·
[Installation](docs/INSTALLATION.md) · [User guide](docs/USER_GUIDE.md) · [All documentation](docs/README.md)

## The easiest way to try Phoenix

DLSS5 Everything is a free, open-source Windows app that gets Phoenix running
in your games. It finds installed games, detects their graphics API and whether
they are 32-bit or 64-bit, and installs the matching Phoenix add-on and runtime
components. You can verify or repair an installation, launch the game from the
library, and restore backed-up files if you want to undo the setup.

It also gives you per-game model and effect settings and can use Phoenix on
windows, the desktop or a webcam. Open the app, pick a game, choose **Install**,
and then **Play**.

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

From DOOM II to Gotham: give familiar worlds another look. These are matching input/output captures from reviewed gameplay replays; click an image for the full-resolution view.

**Batman: Arkham Knight**

| Original input | Phoenix DLSS5 |
| --- | --- |
| ![Batman: Arkham Knight: original input](docs/games/batman_arkham_knight-input.webp) | ![Batman: Arkham Knight: Phoenix DLSS5 output](docs/games/batman_arkham_knight-phoenix-dlss5.webp) |

**Hitman: Absolution**

| Original input | Phoenix DLSS5 |
| --- | --- |
| ![Hitman: Absolution: original input](docs/games/hitman_absolution_x86-input.webp) | ![Hitman: Absolution: Phoenix DLSS5 output](docs/games/hitman_absolution_x86-phoenix-dlss5.webp) |

**DOOM II**

| Original input | Phoenix DLSS5 |
| --- | --- |
| ![DOOM II: original input](docs/games/doom2_dos-input.webp) | ![DOOM II: Phoenix DLSS5 output](docs/games/doom2_dos-phoenix-dlss5.webp) |

[Seven-game comparison gallery, routes and capture details](docs/GAME_GALLERY.md). These earlier captures illustrate the effect; they are not fresh 1.0 Beta game tests.

## Features

- **Pick the look:** Phoenix DLSS5, NVIDIA DLSS5, split views and difference views.
- **Two model sizes:** Original and Lightweight FP8, plus compatible external model sources and a shared model library.
- **Tune live:** tone, structure, skin, masks, intensity and appearance styles.
- **32-bit and 64-bit games:** automatic execution choices with an x64 helper when needed.
- **Stay in the game process when possible:** local D3D11/D3D12 bridges and experimental x86 library routes retain checked helper fallback.
- **Steadier delivery:** bounded DX9 capture overlap, matching input/output frame pairing and continuity checks through focus and viewport changes.
- **Motion you can control:** prefer engine vectors or estimate missing motion, with analysis-resolution and history controls on supported routes.
- **Keep up with driver changes:** experimental x86 bridges discover and validate the installed NVIDIA driver, with results cached per driver and runtime package.
- **Broad capture integration:** D3D9/10/11/12, OpenGL and Vulkan, with route-specific requirements.
- **Build a processing chain:** scaling, color, optional multiscale fusion and motion estimation where supported.
- **Keep comparisons honest:** matching original/modified screenshots, timing history and execution diagnostics.
- **Use it beyond ReShade:** a native SDK and file/framed-stream interfaces.

[Full feature catalog](docs/FEATURES.md) · [Controls and screenshots](docs/USER_GUIDE.md)

**Older RTX kernels exist; older-card validation is still needed.** RTX 20/30/40 paths
are experimental. My NVIDIA hardware testing is on an RTX 5080. Portable CUDA,
Vulkan, HIP, CPU, SYCL and Metal routes have separate limits and performance.
[Compatibility and tested hardware](docs/COMPATIBILITY.md).

## Manual installation

**Native Linux preview:** a per-game Vulkan layer, relocatable model-free package
and optional CUDA backends are available. The SM120 native CUDA executor reuses
the production 154/78-launch model schedules, with bounded byte transport and
explicit hardware validation. 0 A.D. gameplay, timings and remaining limits are
recorded in the [Linux guide](docs/LINUX.md) and [native CUDA guide](docs/LINUX_NATIVE_CUDA.md).

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

Original project code is [Apache 2.0](LICENSE). Third-party components retain their
own terms: [THIRD_PARTY.md](THIRD_PARTY.md) · [NOTICE](NOTICE). Separately supplied
NVIDIA models and runtimes retain their own licenses.
