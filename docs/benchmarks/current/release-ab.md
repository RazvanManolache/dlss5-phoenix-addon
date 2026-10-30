# Historical released/development comparison: the same 310.8 DLL

This is the retained 20 October comparison, including the development build from
that study. The [current native Phoenix DLSS5 / NVIDIA DLSS5 report](README.md)
contains the newer kernel measurements. This controlled release comparison has
not been relabeled as a fresh run.

Measured 20 October 2026 on RTX 5080, driver 616.64. All 40 isolated native inference cases passed. [Machine-readable results, individual process medians and fingerprints](release-ab.json).

## Result

The current source is not uniformly slower than the GitHub source. With the same authenticated 310.8 DLL and input, Phoenix DLSS5 measured 1.32% less GPU time at 720p, 1.05% less at 1080p and 1.46% more at 4K. The 4K increase is a measured difference, not an isolated code regression: clocks were not locked, there are only two processes per build there, and the NVIDIA DLSS5 control also shifted.

Values below are arithmetic means of fresh-process medians, not pooled frames or game FPS. Negative change means less time. No samples or process outliers were removed.

| Renderer | Input | GitHub GPU ms | Current GPU ms | GPU change | GitHub call ms | Current call ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Phoenix DLSS5 | 1280x720 | 2.5291 | 2.4956 | -1.32% | 2.9188 | 2.8147 |
| Phoenix DLSS5 | 1920x1080 | 3.9254 | 3.8844 | -1.05% | 4.3608 | 4.3706 |
| Phoenix DLSS5 | 3840x2160 | 13.3256 | 13.5196 | +1.46% | 13.8685 | 14.0319 |
| NVIDIA DLSS5 | 1280x720 | 2.6234 | 2.6217 | -0.07% | 3.0663 | 3.1376 |
| NVIDIA DLSS5 | 1920x1080 | 3.9272 | 3.9258 | -0.04% | 4.5603 | 4.7033 |
| NVIDIA DLSS5 | 3840x2160 | 13.0532 | 13.3188 | +2.03% | 13.8172 | 14.1049 |

At 1080p, the GitHub Phoenix DLSS5 process medians were 4.028448, 3.890976, 3.891392 and 3.890848 ms; current was 3.862976, 3.886464, 3.895072 and 3.892960 ms. The median of those process medians is 3.891184 versus 3.889712 ms: effectively equal (-0.04%). The apparent 1.05% mean improvement depends heavily on the first released-process result. It is not evidence of a reliable 1% optimization.

At 4K, Phoenix DLSS5 ranges were 13.314880-13.336416 ms released and 13.412256-13.626848 ms current. NVIDIA DLSS5 ranges were 13.005088-13.101344 and 12.982560-13.655136 ms respectively. These ranges describe observations, not confidence intervals. A longer clock-controlled study and kernel-level investigation would be needed to attribute the small Phoenix DLSS5 increase to source changes.

## Older versus newer DLL

The current build also ran Original from each DLL at 1080p, four fresh processes per renderer and DLL.

| Renderer | DLL | Mean process-median GPU ms | Process range ms | Mean process-median call ms |
| --- | --- | ---: | --- | ---: |
| Phoenix DLSS5 | 310.8 | 3.8844 | 3.8630-3.8951 | 4.3706 |
| Phoenix DLSS5 | 310.9.16 | 3.9286 | 3.8789-4.0565 | 4.4045 |
| NVIDIA DLSS5 | 310.8 | 3.9258 | 3.9123-3.9353 | 4.7033 |
| NVIDIA DLSS5 | 310.9.16 | 3.8357 | 3.8316-3.8395 | 4.4263 |

Phoenix DLSS5's medians of process medians were 3.889712 and 3.889536 ms respectively. Its higher new-DLL arithmetic mean is driven by the 4.056544 ms process; these data do not establish a persistent extra inference cost from the newer weight container. NVIDIA DLSS5's newer-DLL GPU interval was about 2.29% shorter on this fixture. Startup, DLL hashing/loading and enable latency are not included in these steady-state intervals.

## Correctness and identities

The exact old DLL was already present locally; its fingerprint was verified and it was used read-only. Neither DLL was replaced or copied into any game.

- 310.8 SHA-256: `e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e`.
- 310.9.16 SHA-256: `2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f`.
- GitHub source: `75ebec5d87deceb0e8f2216fe1d25f5440e4cc42`.
- Current source: `977c2eac7433786ff5fc6593ab756fb4d76ae4c9`.
- Common benchmark frontend SHA-256: `70d4f9c045bce5b314276e214ca35f60cb0fa544bc69a2afb79e96eff092fa21`.

At each tested extent, all reset A, history and reset B output hashes agree across the two source builds and both renderers. The newer DLL's Original also agrees with the older DLL at 1080p. Inputs agree, outputs are finite and resetting reproduces reset A exactly. Each raw result file's fingerprint was rechecked after the run. This is deterministic fixture parity, not moving-scene quality or retail-game compatibility evidence.

## Method and limits

Both committed source trees were exported into disposable folders and rebuilt with CUDA 13.1, including each version's own 41 production kernel modules. One identical Original-only probe was used because the released source predates the new model/schedule API. Only that probe was adapted: each snapshot retained its own production recorder, graph, dispatch, allocation and kernels. The frontend enforces the full eight-transformer network, one pass and all-ones repetitions; no adaptive budget or deep diagnostic markers are enabled.

The GPU-resident structured SDR fixture is quantized to RGBA16F, with zero RG16F motion, fixed seed, automatic mask, default tone/structure and no depth. Each process warms up for one second and measures for three seconds, with at least 30 measured frames. GPU timestamps include engine work, barriers and the matching output copy; input upload/readback is excluded. Call time includes command recording, submission and fence wait. GPU time is inside call time, not additional to it.

There are two fresh processes per build/renderer at 720p and 4K, four at 1080p, plus eight new-DLL 1080p controls: 40 cases total. Cell order reverses every round. GPU telemetry is retained, but clocks, desktop GPU load and OS scheduling are not controlled.

This compares the currently published GitHub source, not commit `c1a6d83b38c64dc5c7e7c6b03008635f63ccbf04` that produced the historical README timings. The earlier table also used a different fixture, debug mode and sampling protocol. Do not splice its numbers into this controlled-input comparison or claim it proves a specific kernel-level cause.

The test covers the native x64 Original model only. It does not benchmark Lightweight, portable providers, x86 transports, addon UI/scheduling, game capture/composition, or startup-to-first-image. No games were launched or installations modified. Temporary builds and fixture buffers were removed after success; compact receipts remain.

## Reproduce

With CUDA 13.1, the NGX headers, MSVC and both authenticated DLLs available, run from the repository root:

```powershell
python tools/benchmark_release_ab.py --released-ref 75ebec5d87deceb0e8f2216fe1d25f5440e4cc42 --old-dll <310.8-dll> --new-dll <310.9.16-dll> --output <fresh-results-folder>
```

The runner compares that released reference with the current committed `HEAD`, verifies both DLL hashes and refuses to merge a run into an existing receipt. Repeating this exact source comparison requires current `HEAD` to be `977c2eac7433786ff5fc6593ab756fb4d76ae4c9`. The Original-only frontend adapter has model-free tests in `tests/python/test_release_comparison.py`.
