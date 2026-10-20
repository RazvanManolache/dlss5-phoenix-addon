# 0.9 Release Validation

Release-package validation is separate from the pre-release performance study.
The downloadable `VALIDATION.json` binds the final archives to their commit,
hashes and checked payloads. No retail game was launched or modified.

| Check | Scope |
| --- | --- |
| Python suites | 117 tests: 33 source/build, 11 contract-tool, 13 publication/privacy and 60 benchmark/report tests |
| Full add-on source build | 21 CPU suites and two software-D3D12 capture/resolution suites |
| x86/x64 model contracts | Selection, synthetic DLL/folder loading, authenticated resource identities, picker and runtime selection; native C ABI on x64 |
| Processing | 65,289 checks on each architecture; persistence, numerical processing and shader compilation, without GPU execution |
| Native kernels | All 41 modules rebuilt; 128 frames at each of eight saved Original-model regression sizes, including both sides of the large-input branch and 8K |
| Current portable references | Both models reproduce the independently saved 20 October reset/history/reset outputs at 320x180 on CPU, CUDA and NVIDIA Vulkan |
| Packaged x64 native library | Eight GPU frames per model at 1080p, with finite output and no debug errors |
| Packaged x86/x64 Vulkan | Nine full frames per architecture/model across 64x64, odd 129x97 and 1080p, with motion, depth and validity; matching complete output bytes across architectures |
| Embedded payload | Both final add-ons and the x86 add-on's embedded x64 core checked recursively as data, with decompression and SHA-256 validation |
| Publication | Source inventory equals the committed tree; archive CRC/content hashes and privacy checks, including UTF-16 strings in Windows binaries |

The portable comparison uses the existing two-model study as its output oracle;
it does not generate new expected hashes from the release candidate. A historical
Original-only `runner_regressions.json` CPU fixture does not match this checkout
or the published 0.82 CPU/runtime DLLs used as a control. Their bytes were checked
against the 0.82 release ZIP. That historical check is not reported as passed or
silently regenerated, and this counterexample is not proof of its root cause.
Its status is retained separately from the passing current references.

These checks do not establish every graphics API's live presentation, VR,
moving-scene quality, HDR behavior, every artistic-control combination, all
hardware, or performance of every x86 transport. The profile-specific native
WOW64 experiments are not freshly certified or bundled in 0.9.

The binary packages contain no NVIDIA model DLLs, weights, SDK/driver DLLs,
game-specific configurations, GPU UUID catalogue, build logs or personal paths.
Historical report device identifiers are redacted without changing their measured
values; parser tests use explicitly synthetic identifiers.
Source-location diagnostics retain mapped project-relative names. The optional
portable package uses `PhoenixRuntime/win-x86` and `PhoenixRuntime/win-x64`
directly, matching the add-on's default lookup paths. It is a runtime package,
not a replacement for building a full CMake development SDK from source.
