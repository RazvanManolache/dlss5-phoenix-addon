# Vulkan fallback optimization — 14 October 2026

The path for GPUs without cooperative matrices now uses a 32×32 shared-memory
matrix tile instead of an 8×16 tile. Four subgroups reuse decoded weights and
activations while each output retains the same ascending FP32 FMA reduction and
half/FP8 publication. Model precision and weights are unchanged.

Expanding packed projections with K ≥ 128, and large 128-to-32 contractions,
use a second compilation of the same source with contiguous weight reads and
padded shared storage. Other shapes keep the simpler tile. Always choosing the
coalesced variant regressed smaller images; shape selection avoids that loss.

These changes apply to the **fallback**, including the Android emulator. The
existing matrix-accelerated desktop path retains byte-identical shader binaries.
Paired desktop measurements found no reliable new speed gain on that path.

## Android game-frame results

Gloomy Dungeons II's autonomous demo, Android 14/API 34 x86_64 emulator 37.2.12,
host RTX 5080/driver 616.64, Ryzen 9 9950X3D, 4 GiB guest RAM and six exposed
logical CPUs. The emulator does not expose cooperative matrices.

| Processing size | Previous call | Selected call | Reduction | Owned model buffers |
| --- | ---: | ---: | ---: | ---: |
| 320×180 | 69.0 ms | 30.1 ms | 56.4% | 180.57 MiB, unchanged |
| 640×360 | 95.7 ms | 45.4 ms | 52.6% | 200.26 MiB, unchanged |
| 1280×720 | 234.0 ms | 121.0 ms | 48.3% | 308.26 MiB, unchanged |

These are synchronous C frame-call medians from ten warm reset calls on a reused
session. They include preparation, model execution, transfers and resolve inside
the library. Initial model loading, the first call, game capture/resizing, PNG
encoding and texture upload are excluded. The input image is identical between
variants, and every output matches the earlier saved RGB32F output byte for byte.
The test still freezes a game frame; it does not measure continuous gameplay FPS.
Emulator timings do not predict physical-phone performance.

Guest peak RSS was approximately 453, 460 and 489 MiB, respectively. These are
process high-water marks including loading transients, not model-buffer sizes or
total host VRAM. This pass does not claim a total-RAM reduction.

## Desktop fallback checks

With cooperative matrices deliberately disabled on the RTX 5080, the initial
full-model sweep reduced warm frame-call medians from 64.1 to 26.3 ms at 320×180,
85.4 to 38.3 ms at 640×360, 207.5 to 100.5 ms at 720p, and 380.1 to 193.2 ms at
1080p. These use a synthetic color input rather than the game frame above.

The available AMD gfx1036 integrated GPU's forced fallback improved from
1,701.9 to 832.8 ms at 320×180, preserving its own baseline output exactly.
This is not a speed claim for its normal matrix-capable route or a discrete AMD GPU.

An ABBA check of the normal RTX accelerated route at 1080p gave approximately
145.3 versus 144.9 ms per call and 107.5 versus 107.1 ms GPU time. That difference
is timing noise, not a claimed acceleration. Existing accelerated shader binaries
are unchanged.

## Correctness and alternatives

- Forty baseline/candidate frame hashes match across 1×1, 96×64, 321×193,
  513×257 and 1920×1080, with accelerated and forced-fallback paths. Cases include
  reset, fractional motion/depth/validity, changed controls and two passes.
- Expanded operator checks cover packed FP8 and ordinary half weights, both
  layouts, partial M/N/K, batch/row strides and output padding. CPU, Vulkan RTX,
  Vulkan AMD and Android Vulkan pass. Each suite also retains 1,354,743 finite
  conversion checks.
- Windows C ABI/platform/operator CTest suites pass; three import-integrity
  checks pass. Imported third-party source files were not modified.
- Optional scalar fused attention uses four 128-thread groups per window/head,
  with sixteen query rows and 6 KiB shared storage per group. It matches the
  separate operations for 1/2/4/8/16 heads on NVIDIA DLSS5, Android and the AMD iGPU.
  Full reset frames also match on Windows/NVIDIA DLSS5. Full-call gains were inconsistent, so
  `vulkan_scalar_attention` remains **false by default**.
- Larger 16×32, 64×32, 32×64 and 64×64 tiles, deeper K staging, unconditional
  coalescing, and an ordered-projection experiment on matrix-capable GPUs were
  measured and retained locally. They were not selected as defaults.

`vulkan_tiled_fallback=false` restores the original tile.
`vulkan_coalesced_fallback=false` retains the simple 32×32 tile everywhere.
Status reports the selected fallback tile and matrix-acceleration availability.

## Reproduction and retained evidence

The ignored workbench is `build/vulkan_next_20261014/`. It retains candidate
sources/presets, provider build recipes, per-operation profiles, full-call timing
samples, hashes, regression logs and model-free Windows/Android x86_64 SDK builds.
`android_comparison.json`, `reference_validation.json`, `selected_validation.json`,
`mixed32_scalar.json`, the `coop_abba_*.json` files and operator logs are the raw
receipts. Android's original baseline captures remain in
`build/android_game_20261014/results/`.

The supplied NVIDIA DLSS5 model DLL remains external. Neither SDK contains the DLL
or extracted model tensors. No physical-phone or discrete AMD/Intel testing was
performed in this pass.
