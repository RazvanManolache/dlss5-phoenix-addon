# Compatibility bridge fixes and regression checks

2026-10-14. Source checkpoint `474e985`. RTX 5080, driver 616.64; Ryzen 9 9950X3D (16 cores / 32 logical threads).

**The reviewed correctness fixes pass the final source build and hardware matrix.** The following benchmarks compare the old embedded helper with the rebuilt helper on the same machine; they exclude game capture/display and are not whole-game FPS measurements.

## Tests

- 15 CPU suites and two software D3D12 suites; packaged sources, objects, resources and complete helper ABI verified.
- 92 IPC CPU checks across x86/x64; cancellation and game-facing lease release, session cleanup failures, handle rollback and retirement ownership.
- 48 rendered shared UI contexts with matching ImGui docking ABI, real Clear Timing clicks, policy/persistence/hotkey checks and nine negative build/exit gates.
- Eight shared/CPU transport checks including 1080p, split matching and helper death; x86/x64 processed pixels agree.
- D3D9, D3D10, D3D10.1, D3D11, D3D12 bridge, OpenGL and Vulkan: seven static, seven changing-input, seven writable-control/exact screenshot checks.
- Two OpenGL windows on the same context with aliased default backbuffers remain independent.
- Both client bitnesses preserve exact legacy FP16 two-pass rounding and reject unequal transport canvases.
- Both bitnesses validate a prospective model with three real CPU frames and reject a separate invalid candidate without activating it.

Real-game follow-up: Sid Meier’s Pirates (32-bit D3D9), addon bridge only; passed. Processed output, disable, Input bypass, resume and screenshot checked; no desktop input.

## D3D11 resize follow-up

Resize testing found a separate stall: destroying the bridge also released its last D3D12 sharing-device reference, which waited for driver/context teardown. The controller now retains only that device for the game's D3D11 device lifetime. Frame textures, views, fences and helper sessions remain independently replaceable. The device owner is cleared on game-device destruction or normal addon detachment.

| Active D3D11 canvas change | Before (ms) | After (ms) |
|---|---:|---:|
| 257×193 → 321×241 | 129.955 | 5.573 |
| 321×241 → 641×361 | 132.161 | 4.688 |
| 641×361 → 257×193 | 136.944 | 4.672 |
| Median | 132.161 | 4.688 |

These three high-resolution timer measurements cover `ResizeBuffers` itself with Phoenix DLSS5 active. Helper initialization and fresh-output waits are outside the timing; this is not an inference-speed improvement or a steady-state FPS measurement. The actual addon supplies the retained device; the measured run uses no external device-retention workaround. The median stall decreased by 96.5%.

The rebuilt package passed all seven static, seven motion, seven control/screenshot and one OpenGL multiwindow checks again. D3D10, D3D10.1, D3D11 and D3D12 each passed the three-size resize cycle, with fresh processed output at every extent and identical pixels on return. The D3D11 test additionally requires sharing-device reuse. Both addon bitnesses compiled; the added x86 resize fixture was compile-checked, while these graphics runtime checks used x64. The core build again passed 15 CPU and two software D3D12 suites.

The original paired helper benchmarks below remain separate measurements of the earlier helper fixes. This resize follow-up does not change the helper's inference or transport algorithms.

## Paired helper benchmarks

Three alternating old/new rounds, 64 completed calls per run, first call excluded from each timing median; the table uses the median of the three run medians. All old/new output hashes match.

| Case | Before (ms) | After (ms) | Change | Helper dedicated GPU peak, before → after (MiB) |
|---|---:|---:|---:|---:|
| Pixel transport 1920×1080 | 4.303 | 3.119 | -27.5% | 44.3 → 44.3 |
| Pixel transport + Phoenix DLSS5 257×255 | 7.587 | 6.660 | -12.2% | 311.5 → 311.2 |
| Pixel transport + Phoenix DLSS5 1920×1080 | 15.430 | 11.322 | -26.6% | 729.5 → 721.5 |
| Pixel transport 3840×2160 | 19.511 | 14.462 | -25.9% | 44.3 → 44.3 |

| Case | Helper private committed memory, before → after (MiB) | Helper working set, before → after (MiB) |
|---|---:|---:|
| Pixel transport 1920×1080 | 179.7 → 181.5 | 161.1 → 161.9 |
| Pixel transport + Phoenix DLSS5 257×255 | 630.8 → 630.0 | 434.4 → 435.0 |
| Pixel transport + Phoenix DLSS5 1920×1080 | 1078.4 → 1066.7 | 441.8 → 442.4 |
| Pixel transport 3840×2160 | 229.7 → 228.1 | 209.7 → 208.5 |

Private committed memory is a process commitment counter; working set reflects resident process pages. These are not interchangeable measures of physical RAM.

Windows process RAM and GPU memory are sampled at 50 ms intervals; the values are observed peaks including startup, not exact allocator totals or steady-state memory. Client and helper shared GPU allocations may appear in both counters; do not sum them. The accompanying JSON contains private bytes, working set and both process GPU counters.

## Changes mapped to the review

| Findings | Resolution |
|---|---|
| R01, R02, R06, R08 | Fence-retired reusable storage, cancellation-aware preparation, shared-owned background workers, per-session nonces, locally owned duplicated handles. |
| R03, R04, R05 | Retire both fallback/shared transports when applying settings; identify presentation windows; writable per-window game-control mailbox. |
| R07, R10, R11 | Lazy iteration scratch, skip unnecessary coherent current-frame copies, cache D3D11 views, reuse upload/readback storage; preserve rare FP16 CPU rounding. |
| R09 | Capture an owned D3D9 snapshot before deferred CPU readback; the driver readback operation remains synchronous. |
| R12–R17, R21 | Shared presets/hotkeys/compact UI, unsupported FPS-profile normalization, acknowledged timing clear, independent candidate validation and explicit helper timing provenance/counters. |
| R18–R20 | Matching docking UI ABI, failure-propagating test harness, full export and source/object/resource packaging verification. |

The final peer pass also made best-effort session cleanup non-throwing. Screenshot capture now stages the explicitly supplied pre-Present image, handles row pitch/BGRA/OpenGL orientation, and encodes after fence retirement.

## Remaining limits

- D3D9 readback still calls a synchronous driver function after snapshot retirement. These helper benchmarks do not prove zero Present stalls under a busy game.
- API runtime tests here use x64 ReShade fixtures. x86 transport, CPU validation and UI checks passed; the previously blocked x86 ReShade fixture was not bypassed.
- Screenshots capture before ReShade effects/UI; HDR/MSAA captures are rejected. Exceptional Vulkan teardown before submission retains resources until device destruction for safety.
- Physical device-removal/fence-failure, forced D3D11 fallback reapply, non-DXGI resize and every GPU vendor are not certified by this hardware matrix.
- NVIDIA vendor-helper NGX initialization remains a separately documented limitation; these results concern the Phoenix DLSS5 compatibility route.
- Candidate validation proves integrity/loading/finite CPU inference, not visual quality or every runner’s GPU compatibility.
- Cleanup of obsolete test binaries was blocked by automatic approval review; binaries remain ignored locally and are absent from the source commit.

## Evidence and reproduction

See [graphics compatibility instructions](GRAPHICS_COMPATIBILITY.md) and [machine-readable summary](BRIDGE_FIXES_2026-10-14.json). Run `test_compatibility.py`, `test_graphics_apis.py`, `test_compatibility_cpu_iterations.py`, `test_model_validation.py`, `test_shared_panel.py`, `test_compatibility_ipc.py` and `benchmark_compatibility.py` with independently supplied SDK/model inputs.

Historical local receipt directory: `build/bridge_fixes_20261014`. Historical package directory: `build/bridge_fixes_acceptance_package_20261014`. These local outputs are not included in the source repository. No model or NVIDIA DLSS5 DLL was embedded.

Resize follow-up receipts: `build/bridge_resize_cached_validation_20261014/report.json` and `build/bridge_resize_verified_20261014/resize_latency_result.json`. The latest rebuilt package is `build/bridge_resize_cached_package_20261014`; its binary hashes are recorded separately in the accompanying summary. It contains no model or NVIDIA DLSS5 DLL and has not yet replaced the real-game installation.
