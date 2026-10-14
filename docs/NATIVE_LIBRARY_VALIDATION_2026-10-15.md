# Native library checkpoint — 15 October 2026

Windows x64; NVIDIA RTX 5080 16 GiB, driver 616.64; Ryzen 9 9950X3D;
96 GiB RAM. MSVC 19.44, Windows SDK 10.0.26100.0. Stock clocks, not locked.
The [compact receipt](NATIVE_LIBRARY_VALIDATION_2026-10-15.json) pins the DLL
hash and measurements. Raw evidence is local under `build/native_library_20261015`.

## Performance

1920×1080 synthetic RGBA16F color/RG16F zero-motion fixture; same authenticated
model, production kernel bundle, all eight blocks once and one inference pass.
Each process runs 128 frames and discards 32 warmup frames. Two processes each
use the original network probe and public texture API in baseline/library/library/
baseline order. Figures below average their process medians; host is one process.

| Boundary | GPU interval | Complete call |
| --- | ---: | ---: |
| Original native network probe | 3.826 ms | 4.191 ms |
| Public native D3D12 texture API | 3.829 ms | 4.262 ms |
| Public native host-buffer API | 3.794 ms | 8.001 ms |

The network probe times inference; the library GPU interval includes inference
and output copy, but excludes input preparation and host transfers. Complete-call
time includes synchronous submission/wait, and transfers for the host route.
These are execution-boundary measurements, not game FPS, a rigorous sub-percent
speedup claim, or measurements of vendor temporal SR. The native path's output
and timing are preserved; host readback has a visible additional cost.

## Correctness and lifecycle

- First/final 1080p output hashes match exactly across both original-probe
  processes, both public texture processes and the host-buffer process.
- A 321×193 fractional-motion fixture also matches original-probe reset/final
  output exactly across 12 frames.
- Twelve frames with 321×193 input, padded 257×193 inference and 643×387 output
  match byte for byte between host and texture APIs. This case varies tone,
  structure and reset and uses block repeats `[1,0,2,1,1,1,1,1]`. All pixels are
  finite and source alpha is retained.
- Actual native 1280×720 inference followed by spatial 1920×1080 output passes
  twelve frames with finite results. This checks execution, not gameplay quality.
- Every real-GPU library probe enables the D3D12 debug layer, rejects error
  messages and verifies no live `ID3D12Resource` remains after library and harness
  destruction. Device-loss/fence-timeout fault injection was not performed.
- Six CTest suites pass: public spatial ABI, C-header consumer, existing addon
  spatial regression, portable C ABI, portable platform and CPU operators.
- Spatial checks cover constant color, source alpha, row-padding guards, 1×1,
  odd canvases, separate larger/smaller output dimensions, numerical four-pixel
  gradient reduction, invalid capacity, texture alias rejection, thread ownership,
  and supplied versus owned devices.
- The existing spatial regression covers gradient error, exact motion scaling,
  padding, alpha, 64 rotating buffers, descriptor reuse and DLL-only construction.

The addon has not been reinstalled or tested in a game for this library change.
The portable ABI remains separate. Native effect composition beyond the controls
listed in [the API guide](NATIVE_LIBRARY.md), additional neural passes and vendor
temporal upscalers are not claimed by this extension.
