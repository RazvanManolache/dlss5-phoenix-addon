# Portable runner optimization follow-up â€” 2026-10-13

This pass keeps exact output while improving four alternate routes. These are
portable-runner measurements, not the native Phoenix DLSS5 or NVIDIA DLSS5 DLL benchmark.
All numbers below use 1920Ã—1080 input and the same external model DLL.

| Route | Previous model time | Selected model time | Model reduction | Previous full call | Selected full call |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUDA, RTX 5080 | 55.687 ms | 51.208 ms | 8.04% | 105.073 ms | 103.340 ms |
| Vulkan, RTX 5080 | 101.453 ms | 97.647 ms | 3.75% | 156.515 ms | 152.269 ms |
| CPU, 32 threads | 9,397.735 ms | 8,182.668 ms | 12.93% | 9,454.281 ms | 8,229.857 ms |
| HIP, AMD integrated GPU | 14,061.468 ms | 13,196.033 ms | 6.15% | 14,767.578 ms | 13,845.771 ms |

The CUDA and Vulkan full-call reductions are **1.65% and 2.71%**. Their model
execution improves more than the complete call: preparation, transfers, resolve
and broker overhead still matter. These backends remain far behind native
Phoenix DLSS5; the CPU and integrated-GPU routes are not interactive at this resolution.

## What changed

- **CUDA window block:** reuse shared storage after its previous contents die,
  normalize Q/K/V in place, and retain the value tile before attention scores
  overwrite its former storage. Compiled shared memory falls from 49 KiB to
  33 KiB per block, allowing three resident blocks instead of two on this GPU.
  The selected kernel uses 76 registers and has no local-memory spills. An
  isolated window block fell from 0.05856 to 0.04169 ms. The original variant
  remains available through `cuda_compact_window_block=false`.
- **Vulkan matrix routing:** use a wider 32-column matrix tile only for shapes
  that won the isolated and full-model tests. Reuse activation fragments and
  stage decoded weights. Other shapes retain the earlier kernel. Explicit
  `vulkan_matrix_columns` overrides remain available; automatic routing uses
  `vulkan_shape_matrices`.
- **Vulkan half conversion:** use hardware conversion on NVIDIA DLSS5, with an integer
  decode that preserves the rounding boundary the driver previously optimized
  away. Preserve special-value behavior. An AMD conversion differed by one ULP,
  so AMD retains the established software path. `vulkan_native_half` controls
  the optimization; the vendor gate applies even when this option is enabled.
- **CPU and HIP precision conversion:** replace repeated software half
  conversion with hardware instructions in the supported implementations,
  preserving rounding and special-value behavior. The CPU implementation uses
  the existing AVX/F16C capability dispatch; older CPUs retain the fallback.
- **Benchmark tooling:** make the isolated Vulkan profiler honor its FP8 matrix
  option, and print the first mismatching matrix coordinate and bits in the
  operator contracts.

## Memory

The owned model allocation remains **453.38 MiB**: 144.51 MiB of weights and
308.88 MiB of scratch. This is host allocation for CPU and device allocation for
the GPU routes. CUDA's shared-memory reduction is a per-block resource saving;
it does **not** reduce that global allocation.

The following are median process samples, not allocator-wide peak measurements.
GPU driver allocations and caller-owned transport are not fully represented by
the owned-model number.

| Route | Previous working set | Selected working set | Previous private commitment | Selected private commitment |
| --- | ---: | ---: | ---: | ---: |
| CUDA | 302.47 MiB | 303.20 MiB | 909.29 MiB | 908.45 MiB |
| Vulkan RTX | 270.62 MiB | 270.48 MiB | 737.36 MiB | 754.54 MiB |
| CPU | 550.21 MiB | 549.87 MiB | 502.12 MiB | 501.23 MiB |
| HIP iGPU | 138.12 MiB | 137.55 MiB | 95.70 MiB | 94.69 MiB |

Private commitment is not resident RAM. Vulkan pipeline/cache commitment varies;
this pass does not claim a reduction in total process RAM or device memory.

## Method and checks

- Hardware: RTX 5080 16 GiB, driver 616.64; Ryzen 9 9950X3D, 16 cores / 32
  threads, 96 GiB system RAM; AMD gfx1036 integrated GPU. CUDA 13.1, sm_120.
  No discrete AMD or Intel GPU was available.
- External model: `nvngx_dlssnr.dll` 310.8.0.0, SHA-256
  `e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e`.
  No model or NVIDIA DLSS5 DLL is included in the addon artifact.
- CUDA/Vulkan: three fresh processes per variant and route, 12 frames per
  process, discard the first two for timing, then take the median of process
  medians. Fractional motion `(0.375, -0.625)`, depth and validity inputs;
  **all 144 output frames match the saved regression hashes exactly**.
- CPU/HIP: ABBA ordering, two fresh processes per variant; each runs two reset
  frames, with the second timed. All 16 reset-frame outputs match exactly.
  These timing measurements do not establish long-sequence temporal performance.
- Final rebuilt runtime: **42 exact reference frames across 18 jobs** on CPU,
  CUDA, HIP AMD, SYCL CPU, Vulkan AMD and Vulkan NVIDIA DLSS5, at 1920Ã—1080, 321Ã—193
  and 96Ã—64. The smaller jobs exercise temporal history. SYCL and Vulkan AMD
  serve as regression routes; no speedup is claimed for them.
- Operator contracts pass on CPU, HIP, CUDA and both Vulkan devices, including
  1,354,743 finite conversion checks, matrix tails/strides, residual publication,
  normalization, attention and feature packing. Forced 32- and 64-column Vulkan
  alternatives also pass the matrix contracts.
- CUDA compact-window Compute Sanitizer: **zero memory errors and zero race
  hazards** in the operator suite.

The portable timing boundary uses RGB32F host/broker input and output. Native
Phoenix DLSS5's separate RGBA16F GPU benchmark has a different boundary; comparing
these full-call values directly would mix transport costs.

## Retained alternatives and local evidence

Larger Vulkan matrix tiles and deeper K staging often regressed. A single generic
wide pipeline saved less than 1 MiB in the paired test and increased model time
from about 97.4 to 100.2 ms. Its `vulkan_specialize_wide=false` override remains.
The experimental FP8 matrix path remains opt-in and disabled by default because
it is still slower than the selected FP16 route. A host-frame conversion
experiment passed contracts but gave no consistent full-call win and was not
selected.

Local, ignored workbench: `build/portable_followup_20261013/`. It retains candidate
sources, build recipes, piece profiles, sanitizer logs and paired measurements.
The authoritative receipts are `final_gpu_ab/report.json`,
`cpu_hip_ab/report.json` and `final_validation/portable_validation.json`.
Selected runtime: `build/portable_followup_runtime_final124/`.
Source commits: `4407ebf` and `118bff1`.

The embedded addon is built from a clean archive of those commits so concurrent
feeder/UI work in the checkout cannot silently alter this candidate. Compilation
and source contract checks are separate from in-game validation.

The addon build passed 11 source test suites and two software D3D12 suites.
Artifact: `build/portable_followup_20261013/addon_source/build/portable_candidate124/phoenix.addon64`,
SHA-256 `e2b061ed62380b6f6ac99b51969daa9316040111e81f473baf1af992f141521c`.
It embeds the validated portable runtime and unchanged native kernel bundle;
the embedded manifest confirms `model_data_included=false`. This candidate has
not had a new live-game test.

Local artifact cleanup reclaimed approximately 5.40 GiB by removing disposable
compiler/linker intermediates and hard-linking unchanged vendor dependency DLLs.
Candidate project DLLs, shaders and configuration remain independent. All
selected-runtime and addon hashes still match their validation receipts after
cleanup. See the local `artifact_cleanup.json` for the exact inventory.
