# Current benchmark snapshot — 12 October 2026

The three PNGs are static image exports of the full local performance report's
measurements, rather than browser screenshots. Their public data snapshot is
[measurements.json](measurements.json); no model weights, captures, vendor
binaries or machine-local paths are included.

## Hardware and boundaries

- NVIDIA RTX 5080, driver 616.64; Ryzen 9 9950X3D, 16 cores / 32 logical workers;
  96 GiB installed RAM; AMD Radeon gfx1036 integrated graphics.
- Native: the same FP16 input/motion, controls and model for Phoenix and NVIDIA
  DLSS NR. GPU timestamps cover the complete runner/Evaluate. Upload and readback
  are excluded; D3D12 debug is enabled for both. Each process uses 128 frames,
  discarding the first 32. Eight fresh processes per route at 1080p; two elsewhere.
- CUDA/Vulkan RTX (12 October): three independent broker processes per route,
  ten frames each, discarding the first two. Eight global blocks, one pass;
  RGB32F/depth fixture with fractional motion (+0.375, -0.625) and history
  validity. Default grouped GPU timing. Headline values are medians of process
  medians. Other portable routes retain their 11 October timings: two processes
  per route, three measured history frames each. All rows show measurement dates.
- Complete portable frame time includes CPU preparation/history, transfers,
  inference, resolve and broker round trip. Startup and file I/O are excluded.
  Model GPU time is part of that boundary. Native and portable fixtures differ.
- The current CUDA/Vulkan graphs allocate 475.26 MiB: 144.51 MiB packed weights
  and 330.75 MiB scratch/activations. Worker RAM is the median of the final-frame
  working-set observations across processes, not peak RAM or total VRAM.
- Native comparison and full-adapter memory observations remain dated 11 October;
  those implementations were unchanged in this portable optimization pass.
- Memory: process GPU-local residency includes harness and driver allocations;
  owned resources count the runner's allocations. Worker RAM is resident working
  set, not private commitment. These scopes overlap and must not be summed.
  Integrated GPU allocations use shared RAM. NVIDIA internals are opaque.
- The full-adapter breakdown is a separate test from the native inner-network
  comparison. At 1080p it owns 493.94 MiB for one pass, 527.69 MiB for two,
  and 595.19 MiB for four. Returning to one pass restores 493.94 MiB owned;
  driver residency can retain caches. Tracked owned allocations reach zero at
  shutdown; capture allocations retire after use.

## Native comparison

| Resolution | Phoenix GPU ms | NVIDIA DLSS NR GPU ms | Phoenix latency difference |
| --- | ---: | ---: | ---: |
| 640 × 384 | 1.9364 | 2.0153 | -3.92% |
| 1280 × 720 | 2.5049 | 2.5912 | -3.33% |
| 1920 × 1080 | 3.8351 | 3.8627 | -0.71% |
| 2560 × 1080 | 4.8340 | 4.8096 | +0.51% |

At 1080p the matched probe's GPU-local residency was 574.70 MiB for Phoenix and
686.52 MiB for NVIDIA: 111.83 MiB / 16.29% lower. This is an observed process
comparison, not a claim to know NVIDIA's weight/workspace sizes or every game's
memory saving. Small timing differences should be read as near parity.

## Checks performed

- 2,688 native timed frames; four reset-output checks plus final native/NVIDIA
  output comparisons, matching on the fixed fixture at all four extents.
- 42 saved-regression frames across CPU, CUDA, HIP, SYCL CPU, Vulkan AMD and
  Vulkan NVIDIA on the current runtime. The latest CUDA/Vulkan matched run
  checked 120 exact temporal frame hashes; the current candidate contributes
  60 frames, including 48 measured frames after warmup.
- Fused residual/attention, row normalization, frame packing and conversion
  contracts passed. Native CUDA conversions matched 4,964,336 reference checks.
- Current portable source checkpoint: `5c0bca5`. The new addon passed 11 CPU
  suites, two software D3D12 suites and all 120 embedded-file checks. This exact
  addon binary has not received a new live-game test.
- Full-adapter lifecycle/capture retirement checks and separate Vulkan allocator
  observations. No claim that portable and native outputs are bit-identical.
- No live-game FPS result, discrete Radeon result, Intel GPU result or measured
  dual-discrete-GPU speedup is implied by this snapshot.

## Regenerate the images

From the repository root, with Python and the optional Pillow documentation tool:

```powershell
python -m pip install pillow
python tools/render_readme_benchmarks.py --snapshot docs/benchmarks/2026-10-12
```

This redraws the checked-in snapshot; it does not rerun inference or need a model.
The addon build does not require Pillow. Colors are consistent with the local
HTML report. The charts use Segoe UI on Windows, with DejaVu Sans as a Linux fallback.

To refresh aggregate portable data from a verified local run, use
`python tools/update_readme_portable_snapshot.py --help`. It publishes aggregate
measurements and evidence digests, without local model/runtime paths or captures.
Historical snapshots remain available for reproducibility; the README shows the
current snapshot only.
