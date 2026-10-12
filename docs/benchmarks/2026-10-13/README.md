# Current benchmark snapshot — 13 October 2026

Native RTX is the project's primary execution path. The alternate APIs are
experimental portability work and are reported separately. This snapshot shows
current measurements only; it does not imply a new native speedup.

The PNGs are static chart exports, not browser screenshots. The aggregate source
is [measurements.json](measurements.json). No models, captures or vendor binaries
are included. Snapshot date is 13 October; actual measurement dates are below.

## Native RTX and NVIDIA DLSS NR — measured 11 October

RTX 5080, NVIDIA driver 616.64. Identical synthetic FP16 input/motion, controls
and model. GPU timestamps cover the complete runner/Evaluate, excluding upload
and readback; D3D12 debug is enabled for both. Each process runs 128 frames,
discarding 32. Eight fresh processes per route at 1080p; two elsewhere.
The native kernel bundle is unchanged in this release.

| Resolution | Phoenix GPU ms | NVIDIA DLSS NR GPU ms | Phoenix latency difference |
| --- | ---: | ---: | ---: |
| 640 × 384 | 1.9364 | 2.0153 | -3.92% |
| 1280 × 720 | 2.5049 | 2.5912 | -3.33% |
| 1920 × 1080 | 3.8351 | 3.8627 | -0.71% |
| 2560 × 1080 | 4.8340 | 4.8096 | +0.51% |

Small differences mean near parity, not game-FPS gains. The matched 1080p probe
observed **574.70 MiB** process GPU-local residency for Phoenix and **686.52 MiB**
for NVIDIA: **16.29% lower**. Residency includes harness and driver allocations;
NVIDIA's internal allocation breakdown is unknown.

## Experimental alternate runners — measured 12 October

1920×1080 active, 1920×1088 padded; full model, one pass, all eight repetition
controls set to one, 512-window batch limit. RGB32F/depth fixture with fractional
motion (+0.375, -0.625) and temporal history. Startup and file I/O are excluded.
Full call includes preparation/history, transfers, model execution, resolve and
broker round trip. This differs from the native fixture/boundary, so a direct
speedup ratio is inappropriate. These are offline timings, not game FPS.

- RTX: three independent processes × 20 frames, discard two per process.
- CPU/SYCL CPU/HIP/Vulkan AMD: two × three, discard one per process.
- Headline values are medians of process medians. CPU K128 timing comes from
  build 078; final 079 centralizes the identical dispatch predicate. HIP uses
  the paired 077 receipt; that implementation is unchanged in the final assembly.
- The AMD Vulkan headline uses process medians like the other rows. Taking the
  median of all four warm frames instead yields 14,246.90 ms; the small difference
  is aggregation, not a different build or a new performance claim.
- Model timing uses GPU events on GPU routes and host timing on CPU routes.

| Runner / hardware | Model ms | Full call ms | Requested buffers MiB | Worker working set MiB | Worker private commit MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUDA C++ · RTX 5080 | 67.37 | 112.04 | 453.38 | 315.78 | 927.37 |
| Vulkan C++ · RTX 5080 | 111.31 | 161.68 | 453.38 | 269.42 | 688.55 |
| CPU C++ | 9,107.65 | 9,167.84 | 453.38 | 581.54 | 501.33 |
| HIP C++ · AMD iGPU | 14,068.66 | 14,793.48 | 453.38 | 169.17 | 94.60 |
| Vulkan C++ · AMD iGPU | 14,142.86 | 14,240.38 | 453.38 | 640.42 | 546.11 |
| SYCL C++ · CPU | 9,867.08 | 9,928.04 | 453.38 | 795.21 | 954.28 |

Test hardware: Ryzen 9 9950X3D, 16 cores / 32 logical workers, 96 GiB RAM;
RTX 5080, driver 616.64; integrated Radeon gfx1036 (PCI 1002:13c0), driver
32.0.21045.5002. SYCL was exercised through Intel's CPU runtime on the Ryzen,
not an Intel GPU. Discrete AMD and Intel GPUs remain untested.

Requested buffers are **453.38 MiB**: **144.51 MiB** packed weights and
**308.88 MiB** scratch/activations. They exclude host planes, caller transport,
driver/JIT caches and other process allocations. Working set is resident worker
RAM; private commit is committed private virtual memory. Both are medians of
final-frame observations across processes, not peak measurements. These scopes
overlap and must not be summed. iGPU allocations use shared system memory.

## Native adapter memory — measured 11 October

The full game adapter owns **493.94 MiB** at one pass, **527.69 MiB** at two and
**595.19 MiB** at four. Model and scratch workspace are shared across serial
passes; each extra pass adds 33.75 MiB history. Returning to one restores
493.94 MiB; driver residency may retain caches. Tracked owned allocations reached
zero at shutdown and capture readbacks retired. This scope includes bridge and
exposure resources; the native inner-network probe owns 409.69 MiB.

## Validation and release

- Native: 2,688 timed frames, four reset checks and final native/NVIDIA comparisons
  matching on the fixed fixture at all four extents, retained from 11 October.
- Final portable assembly: **42 exact saved frames / 18 jobs / six routes**.
- Each latest RTX paired trial: **120 exact temporal frame hashes**, 60 from the
  selected candidate, including 54 candidate frames after warmup.
- Six expanded operator suites cover matrix publication, tails, strides, rows
  and feature packing; 1,354,743 finite conversion checks per backend passed.
- Final addon: **11 CPU suites, two software D3D12 suites, 120 embedded files
  verified**. No model or NVIDIA DLL embedded. This exact binary has not had
  a new live-game test. No portable/native bit-equivalence claim is made.
- Addon source checkpoint: `85b0b82488dc4d1a9cbad069e70c8682c9829791`.
  Documentation may have later commits. Addon SHA256:
  `938e80922a6f7bfa0824a7a6b7a3fb75635301f2174db6ebbc0bb147a167955e`.
- No live-game FPS, discrete AMD/Intel GPU or dual-discrete-GPU speedup is implied.

## Regenerate the charts

From the repository root:

```powershell
python -m pip install pillow
python tools/render_readme_benchmarks.py --snapshot docs/benchmarks/2026-10-13
```

This redraws this snapshot and needs no model. Pillow is documentation-only.
Fonts use Segoe UI on Windows with DejaVu Sans as a Linux fallback.
`tools/update_readme_portable_snapshot.py --help` describes refreshing aggregates
from verified receipts; repeat `--measurements` for separate route trials.
Public evidence digests retain no local paths or captures. Historical snapshots
remain available; the main README displays this current snapshot only.
