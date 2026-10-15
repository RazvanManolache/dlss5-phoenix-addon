# Optional mobile Vulkan compatibility — 15 October 2026

For the separately selected approximate path, see [Fast GPU arithmetic](VULKAN_FAST_MOBILE.md). The exact compatibility measurements below remain unchanged.

The portable library accepts `vulkan_workgroup_compatibility: true` for devices
such as the tested Adreno 735 (SM8635, Android API 37), whose controlled subgroup
range is 64–128 rather than 32. The option defaults off. Standard and optional
capability queries remain separate; unsupported baseline compute features still
fail explicitly. No CPU fallback is reported as Vulkan execution.

## Implementation

- Integer half storage preserves subnormals on this driver. Selected FP32 fused
  arithmetic, square root and division use integer significands to preserve
  publication/normalization boundaries.
- Matrices use 4×4 register blocks within 32×32 workgroup tiles, retaining
  ascending reduction/FMA order for every output.
- Eligible attention rows use eight local invocations and shared-memory
  exchanges. Partial rows join every barrier and mask their writes. No hardware
  subgroup width or subgroup-to-local mapping is assumed.
- Subgroup-dependent matrix, attention and QKV fusions are disabled in this
  explicitly selected route. The 40 existing standard shaders are unchanged.
- The runner supports `benchmark_iterations` (1..100) and detailed operator-kind
  profiling. Repeated-input measurements reset history each iteration.

## Physical-device checks

The Adreno phone passed 1,354,743 finite conversion checks, 2,135,040
normalization channels, matrix tails/layouts/strided batches, local/global
softmax, feature packing, and every full-model checkpoint against CPU.
A 320×144 museum fixture uses a padded 320×320 working surface. All eight
block repetition counts were one, with one pass and reset history.

CPU and both Vulkan output hashes are identical:
`8c6bb14d3e2198dadf68f314ba44afd1a52a633d5f4bba582d338c6f020d6eee`.
This validates this fixture/device and the listed contracts, rather than all
mobile GPUs or arbitrary model controls and temporal sequences.

## Performance boundary

Two serial headless processes use the same input, native libraries and controls;
only the compatibility matrix/parallel-row shaders differ. Each runs seven
calls, excludes the first setup call, and reports medians of six warm calls.

| Measurement | Earlier compatibility shaders | Optimized shaders | Speedup |
| --- | ---: | ---: | ---: |
| Complete C frame call | 2009.685 ms | 1247.084 ms | 1.61× |
| GPU submission spans | 1882.326 ms | 1101.168 ms | 1.71× |

Warm calls range from 1872–2213 ms before and 1121–2035 ms after. Cold calls
still take about 30–31 seconds including graph/pipeline setup. These results
do not establish real-time inference or sustained thermal/battery behavior.
The [machine-readable receipt](VULKAN_MOBILE_2026-10-15.json) records the samples.

A separate local museum harness confirms processed pixels reach the physical
display. Eight screen-on calls have a median 1464.125 ms, about 0.655 completed
calls/s, alongside 12.24 display FPS. Whole-GL checks before/after touch agree
within one byte level; an Android compositor scene strip agrees exactly.
These are application measurements, separate from the SDK timing above.
The museum APK and its external model/assets are not part of this SDK release.

## Build and use

Follow [the library build guide](LIBRARY.md#enable-vulkan-on-linuxandroid).
glslang must include its optimizer (`ENABLE_OPT=ON`) and SPIRV-Tools dependencies.
The native Windows RTX extension remains a separate
[D3D12/NVAPI API](NATIVE_LIBRARY.md); these mobile scalar timings do not describe it.
