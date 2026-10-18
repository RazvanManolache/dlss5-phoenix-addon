# Regular performance regression checks — 14 October 2026

**The regular benchmark suite and exact-output checks passed on the current addon payload.** This follows the separate transport/helper and graphics-bridge checks; their timing boundaries differ.

Hardware: RTX 5080 16 GiB, driver 616.64; Ryzen 9 9950X3D, 16 cores / 32 logical threads; 96 GiB system RAM; AMD integrated gfx1036 for HIP/Vulkan AMD. Model DLL 310.8.0.0 supplied externally. All runs use the full network and one pass. No NVIDIA DLSS5 DLL or extracted model is embedded or published.

The tested addon is `build/bridge_resize_cached_package_20261014/Phoenix.addon64`, SHA-256 `c04132e71517be5743cf3250210a39d7c332d8e450eb6b29a7ae4c6880253c6d`. All 127 embedded payload files were decoded and checksum-verified directly from that addon. The native kernel manifest and every portable runtime/shader binary match the previous benchmark package exactly. The native probe and portable client were rebuilt from current source.

## Native Phoenix DLSS5 and NVIDIA DLSS5 NR

Same synthetic RGBA16F color and RG16F zero-motion input, controls and model. PNNP ordering: two fresh processes per renderer/resolution, 128 frames each, discarding 32. Each table value is the arithmetic mean of the two process medians. GPU timestamps exclude uploads/readbacks. A separate NVIDIA DLSS5 reset probe checks the first-frame hash. All nine reset/final comparisons and the saved-gallery hashes match.

| Resolution | Phoenix DLSS5 GPU ms | NVIDIA GPU ms | Phoenix DLSS5 GPU MiB | NVIDIA GPU MiB |
|---|---:|---:|---:|---:|
| 640×360 | 1.922 | 2.007 | 295.01 | 438.21 |
| 854×480 | 2.088 | 2.159 | 318.70 | 460.40 |
| 960×540 | 2.126 | 2.228 | 339.88 | 480.15 |
| 1280×720 | 2.497 | 2.589 | 395.13 | 532.21 |
| 1920×1080 | 3.829 | 3.864 | 558.82 | 686.52 |
| 2560×1440 | 6.262 | 6.152 | 759.01 | 872.65 |
| 3840×2160 | 13.942 | 13.628 | 1367.20 | 1448.46 |
| 5760×3240 | 32.923 | 32.513 | 2765.13 | 2765.90 |
| 7680×4320 | 62.336 | 61.039 | 4686.51 | 4562.52 |

GPU MiB is whole-process local residency, including the probe and driver. Residency is unchanged at each extent compared with the previous measurements. Host memory counters are retained in the JSON; the two probes retain different host caches, so these are not matched runner-only RAM measurements.

## Alternate runners at 1080p

Saved RGB32F broker fixtures and controls were replayed with exact hashes on every frame. CUDA and Vulkan RTX use three processes with 12 fractional-motion frames each, discarding two. CPU/HIP use two processes with one warmed reset frame each. Vulkan AMD and SYCL CPU use two processes with two measured fractional-motion frames each. The table uses the median of process medians. Full call includes preparation, transfers, execution, resolve and broker overhead; startup, file I/O and output hashing are excluded.

| Runner / hardware | Model ms | Full call ms | Owned graph MiB | Worker working set MiB |
|---|---:|---:|---:|---:|
| CUDA · RTX 5080 | 56.31 | 102.84 | 453.38 | 303.34 |
| Vulkan · RTX 5080 | 104.91 | 158.07 | 453.38 | 313.52 |
| CPU · Ryzen 9950X3D | 9311.58 | 9363.97 | 453.38 | 548.77 |
| HIP · AMD integrated gfx1036 | 14104.37 | 14784.61 | 453.38 | 137.71 |
| Vulkan · AMD integrated gfx1036 | 11999.67 | 12105.86 | 453.38 | 685.91 |
| SYCL · Ryzen CPU / Intel CPU runtime | 11291.05 | 11370.68 | 453.38 | 794.62 |

CPU used all 32 workers. CPU/SYCL operator time uses host timing; GPU operator time uses GPU timestamps. Owned graph memory is requested weight/scratch storage, not whole-process GPU residency. Worker working set, private commitment and graph allocations overlap and must not be added. Owned graph memory is unchanged for every route. These alternate fixtures and timing boundaries differ from native; these timings are not game FPS.

## Earlier/current executable controls

The larger native timings and several portable timings differ from the 13 October snapshot. Identical model/kernel/runtime binaries rule out a new algorithm change in the packaged runners. To check the rebuilt measurement tools, the earlier and current executables were interleaved under today’s conditions, using the same payload, inputs and controls. Native runs use 128 frames with 32 warmup frames; portable runs use the same saved sequences as above. 4K and portable values are means of two process medians per executable. The first earlier-executable 8K process had an unexplained 403.262 ms median; its next process returned to 60.998 ms. An extra balanced repeat reproduced the split: 405.141 ms and 59.753 ms for the earlier executable, while all four current-executable controls remained between 59.144 and 60.425 ms. All observations are retained; the earlier 8K controls have two fast and two stalled processes, so a single percentage would be misleading. All controls pass exact-output checks.

| Control | Earlier executable ms | Current executable ms | Change |
|---|---:|---:|---:|
| Native GPU · width 3840 | 13.709 | 13.717 | +0.06% |
| Native GPU · width 7680 | 59.753–405.141; two stalled | 59.144–60.425 | Baseline unstable |
| CUDA · RTX 5080 | 100.683 | 101.447 | +0.76% |
| Vulkan · RTX 5080 | 155.704 | 154.597 | -0.71% |
| CPU · Ryzen 9950X3D | 9469.614 | 9135.733 | -3.53% |

Native controls report GPU intervals; portable controls report full calls. Stock clocks and system activity were observed rather than locked. Differences within this small sample are observations, not guaranteed speedups or statistical equivalence. The original charts retain their original measurement date.

## Checks and build repair

- 36 timed native processes and nine reset probes: 4,617 executed frames, 3,456 measured frames, nine standard extents from 360p through 8K.
- 14 alternate-runner processes across all six routes: 92 exact-output frame comparisons.
- Twelve interleaved native controls and 12 portable client controls, all with exact outputs; both earlier-executable 8K stalls remain recorded.
- The offline validation-tool builder still requested C++17 although the current model loader requires C++20 (`std::span`). It now selects C++20 and the matching language-version macro. All four validation targets compile; the current native probe executes successfully.
- This is an offline normal-runner regression check; it does not measure live-game capture/display, cross-GPU transfers or every GPU vendor.

Full local receipts and exact replay jobs: `build/regular_performance_20261014`. [Machine-readable measurements and evidence hashes](REGULAR_PERFORMANCE_REGRESSION_2026-10-14.json). [Separate bridge regression results](BRIDGE_FIXES_2026-10-14.md).
