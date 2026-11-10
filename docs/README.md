# Phoenix DLSS5 documentation

[Project overview](../README.md) · [Download 1.1.0](https://github.com/RazvanManolache/dlss5-phoenix-addon/releases/tag/1.1.0)

## Start with your goal

| Goal | Start here | Details |
| --- | --- | --- |
| Install or update | [Installation](INSTALLATION.md) | [Package contents and validation](RELEASE_1.1.0.md) |
| Develop native Linux support | [Linux layer preview](LINUX.md) | Per-game launcher, packaging, limitations and validation |
| Try the Windows CUDA preview | [Windows CUDA integration](WINDOWS_CUDA.md) | Optional native executor, packed transport, build and validation scope |
| See it in actual games | [Seven-game comparison gallery](GAME_GALLERY.md) | [Capture method](PAIRED_SCREENSHOTS.md), [compatibility](COMPATIBILITY.md) |
| Change the image or overlay | [User guide](USER_GUIDE.md) | [All features](FEATURES.md), [appearance controls](NEURAL_CONTROLS.md), [overlay reference](overlay-ui.md) |
| Pick a model | [Model support](MULTI_MODEL.md) | [Shared model library](SHARED_MODELS.md), [Lightweight FP8](SMALL_FP8.md) |
| Understand x86 or older GPUs | [Compatibility](COMPATIBILITY.md) | [Execution mechanisms](EXECUTION_MECHANISMS.md), [graphics APIs](GRAPHICS_COMPATIBILITY.md), [native architectures](NATIVE_ARCHITECTURES.md) |
| Compare speed or memory | [Performance guide](PERFORMANCE.md) | [Current measured results](benchmarks/current/README.md), [memory ownership](../MEMORY_USAGE.md) |
| Build from source | [Addon build recipes](DEVELOPMENT.md) | [Toolchain setup](../BUILDING.md), [external dependencies](../third_party/EXTERNAL_DEPENDENCIES.md) |
| Integrate the runtime | [Library guide](LIBRARY.md) | [Native library](NATIVE_LIBRARY.md), [feeder API](FEEDER_API.md) |
| Contribute or find attribution | [Project background](PROJECT.md) | [Upstream credit ledger](UPSTREAM_CREDITS.md), [third-party terms](../THIRD_PARTY.md) |

## Focused guides and investigations

- [Processing chain](PROCESSING_CHAIN.md) and [pipeline implementation](PIPELINE_IMPLEMENTATION.md).
- [Frame delivery and worker overhead](FRAME_DELIVERY_OPTIMIZATION.md): bounded capture overlap, frame identity and transfer-stage measurements.
- [Paired screenshots](PAIRED_SCREENSHOTS.md) and the optional [Phoenix Input companion](PHOENIX_INPUT.md).
- [Resolution scaling profile](NATIVE_SCALING_PROFILE.md): pixel growth, padding/token boundaries and kernel attribution.
- [Native instruction tuning](NATIVE_TUNING_RESOLUTIONS.md): measured changes across 720p–8K and older-device limits.
- [Portable reuse study](PORTABLE_REUSE_STUDY.md): CPU/Vulkan improvements and rejected CUDA experiments.

Historical reports retain their dates and measurement scopes. Source support,
published package contents, isolated tests and game validation are distinct;
use the linked support/validation notes when choosing a route.

- [Native CUDA for Linux](LINUX_NATIVE_CUDA.md): production model schedule, bounded byte transport and explicit hardware checks.
