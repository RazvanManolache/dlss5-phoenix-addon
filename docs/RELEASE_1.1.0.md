# Phoenix DLSS5 1.1.0

- Native CUDA approaches native RTX/NVIDIA processing speed on the tested RTX 5080, with Original and Lightweight FP8 support.
- Linux gains a per-game Vulkan layer, CPU/CUDA/Vulkan processing, a relocatable SDK and an optional SM120 native CUDA executor.
- Windows gains persistent temporal state and extended guide, motion, mask and appearance controls on supported native CUDA routes, including shared D3D12 textures.
- Portable CUDA, Vulkan and CPU execution benefit from reduced host overhead, reuse and transport improvements.

**2,219 validation checks passed; tested across 76 games.**

DLSS5 Everything is the easiest way to try Phoenix: it is free and open source,
finds games, installs the matching components and manages per-game settings.
[Download Everything](https://github.com/RazvanManolache/DLSS5-Everything/releases/latest).

## Downloads

Use **Windows.zip** for the 32/64-bit ReShade add-ons, graphics helper and optional
new x64 CUDA/Vulkan runtime. Install only the matching add-on. The included
`PhoenixRuntime/win-x64` folder can be selected in the portable runtime settings;
see [Windows CUDA setup](WINDOWS_CUDA.md). The normal embedded native RTX path
remains available. **Input-Windows.zip** contains the optional input companion.
**Portable-Runtimes.zip** supplies the separate x64/x86 runtime packages.

Use **phoenix-linux-x86_64.tar.gz** for Linux. It includes the per-game
Vulkan layer, launch tools, CPU/Vulkan/CUDA providers, native CUDA kernels and
SDK. See [Linux setup](LINUX.md) and [native CUDA boundaries](LINUX_NATIVE_CUDA.md).
**Source.zip** contains the complete committed source tree. Model weights and
NVIDIA model DLLs are obtained separately and are absent from every package.

## Measured performance

| Processing route | Original ms | Lightweight FP8 ms |
| --- | ---: | ---: |
| Native CUDA | 4.63 | 3.42 |
| Native RTX / D3D12 | 4.48 | 3.19 |
| NVIDIA reference | 4.36 | 3.03 |

These are warmed 720p host processing measurements on RTX 5080, including
upload, inference and readback: 300 warmup frames and 60 measured samples per
process, repeated in three fresh processes. Native CUDA is approximately 6%
and 13% behind the reference here. They are processing latency, not game FPS.
Isolated 1080p GPU inference remains near parity: native RTX 3.837/2.249 ms,
NVIDIA 3.846/2.246 ms for Original/FP8.

See [validation details](VALIDATION_1.1.0.md) for coverage and remaining limits.
