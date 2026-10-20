# Runtime and memory review fixes

Measured 19 October 2026 against local checkpoint `bdc3fd8` on RTX 5080 / driver 616.64. These are isolated probes, not game FPS or deployment validation. No games or installed files were changed.

[Compact measurements](review-fixes.json)

## Changes

- Vulkan fused feed-forward is disabled by default after its documented 4K reset failure. Explicit opt-in remains experimental; the faulty optimized shader itself is not repaired.
- Portable schedules reuse immutable uploaded weights. A schedule change still rebuilds commands/scratch and resets incompatible history, but does not re-read/decode/upload already cached weights.
- Newly enabled, previously uncached transformers can decode weights once. The cache is scoped to one backend/model and releases on model replacement or session destruction.
- The addon releases packed CPU weights after session construction. A later session/device recreation re-authenticates and reloads them on demand; this trades occasional setup work for lower steady RAM.
- DLL hashing uses read-only file mappings rather than whole-file private byte vectors. File-backed pages can still affect the working set; this is not a promise of zero RAM during hashing.
- Appearance mask, composition and distortion resources allocate separately on first use. Enabling the first mask retires prior GPU work before publishing its descriptor. This is a one-time synchronization, not a per-frame restart.
- Native schedule metadata and matched NVIDIA presets belong to model definitions rather than a two-model boolean. New architectures still require their own validated graph/decoder/kernel support.

## Schedule changes

CUDA host-plane probe, 320x180, depth sequence 8 -> 4 -> 8 -> 4 -> 8. Median wall time of the four changed-schedule calls includes inference, transfers, graph setup and CPU work; the initial cold frame is excluded. A separate reset/history/reset sequence and protected-mask check run for each model.

| Model | Before ms | After ms | Reduction |
| --- | ---: | ---: | ---: |
| original | 1945.23 | 18.58 | 99.0% |
| small-fp8 | 1762.31 | 10.06 | 99.4% |

## Appearance allocations

Actual D3D12 allocation-info bytes of the addon appearance helper, not raw pixel estimates or whole-process residency. Resources stay alive during each snapshot and return to zero live tracked textures after destruction. The probe records but does not submit its command lists. All values below are MiB.

| Extent | Unused before / after | Style-only before / after |
| --- | ---: | ---: |
| 1920x1080 | 33.75 / 0.00 | 160.31 / 67.50 |
| 3840x2160 | 127.50 / 0.00 | 605.62 / 255.00 |

The removed permanent CPU packed-weight payload is approximately 141 MiB for Original and 109 MiB for Lightweight. This is a source-confirmed ownership change, not a measured whole-game RAM delta: the standalone native probe already freed that copy before these fixes.

## Portable process RAM

Fresh-process CUDA snapshots at 320x180, after the same five-call sweep, temporal check and zero-mask check. The peak is the process-lifetime working-set high-water mark, including DLL/provider initialization. The process also includes Python/NumPy, the driver and allocator caches. All values are MiB.

| Model | Working set before / after | Peak working set before / after | Private commit before / after |
| --- | ---: | ---: | ---: |
| original | 212.3 / 212.4 | 597.7 / 469.7 | 1393.6 / 1504.6 |
| small-fp8 | 203.4 / 203.7 | 590.9 / 441.7 | 1323.9 / 1459.5 |

Startup peaks improve, but steady resident RAM is essentially unchanged and private commit is not uniformly lower. The after-version additionally alternates depth fifty times: tracked backend allocations and GPU-local usage remain constant. In these stress samples, private commit warms by about 77 MiB over the first ten changes and then plateaus through change fifty, while the working set stays nearly constant. The extra untracked commit is not attributed to a specific allocator/driver component; this bounded result is not proof against every long-session leak.

## Native inference

One fresh process per cell and version, 0.5 s minimum warmup and 1 s minimum sampling, with at least 30 measured frames. GPU timestamps cover native engine work and matching output copy, excluding input upload/readback. Clocks are not locked and these single-process medians do not establish small performance regressions or improvements.

| Renderer | Model | Extent | Before GPU ms | After GPU ms |
| --- | --- | --- | ---: | ---: |
| NVIDIA DLSS5 | original | 1920x1080 | 3.784 | 3.785 |
| NVIDIA DLSS5 | original | 3840x2160 | 12.404 | 12.386 |
| NVIDIA DLSS5 | small-fp8 | 1920x1080 | 2.207 | 2.205 |
| NVIDIA DLSS5 | small-fp8 | 3840x2160 | 7.182 | 7.114 |
| Phoenix DLSS5 | original | 1920x1080 | 3.852 | 3.853 |
| Phoenix DLSS5 | original | 3840x2160 | 13.178 | 13.181 |
| Phoenix DLSS5 | small-fp8 | 1920x1080 | 2.287 | 2.282 |
| Phoenix DLSS5 | small-fp8 | 3840x2160 | 7.485 | 7.526 |

Native reset/history/reset outputs match before/after and the corresponding NVIDIA DLSS5 fixture byte for byte at 1080p and 4K for both models. Portable reset outputs also match before/after; CUDA history/reset and zero-mask protection pass. Vulkan 4K passes repeated resets with the new default and no caller workaround. This does not establish moving-game/HDR parity or universal Vulkan correctness.

## Validation and limits

- 21 x86/x64 test/build cases pass, including resident-weight ownership, zero re-uploads for cached schedules, lazy addition of unseen weights, backend isolation, native bundle rehydration, original graph fingerprints, controls, software D3D12 captures and both frontend links.
- Existing fingerprints were not regenerated. Source-independent before/after output comparisons remain required after numerical changes.
- Process RAM/residency snapshots are retained in the compact JSON. Private commit, working set, GPU-local residency and requested allocation bytes are different metrics; do not add them.
- The earlier broad two-model tables remain historical. These changes do not establish that Phoenix DLSS5 uses less VRAM than NVIDIA DLSS5 at every resolution; the 4K scratch-space difference still exists.
- No long-running game, VR, multi-GPU, Metal, HIP or SYCL device validation was performed for these fixes. No release or installed game was updated.

## Reproduce

```powershell
python tools/test_review_fixes.py
python tools/benchmark_review_fixes.py before
python tools/benchmark_review_fixes.py after
python tools/report_review_fixes.py
```

The before probe extracts `git archive bdc3fd8` into its temporary build; use `--baseline-ref` to select another checkpoint explicitly. Dependencies are external local model/kernel/provider artifacts; the probes do not download or modify them. Temporary executables and inputs are removed on success or failure.
