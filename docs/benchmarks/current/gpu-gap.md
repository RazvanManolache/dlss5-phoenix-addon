# Investigating the nearly equal 1080p GPU gap

Measured 20 October 2026 on RTX 5080 / driver 616.64; native Windows x64 / D3D12. This is an isolated inference diagnostic, not game FPS or a compatibility claim.

**Verdict:** the roughly 0.05-0.06 ms Phoenix DLSS5 gap to NVIDIA DLSS5 at 1080p reproduces for both models, but is not a universal fixed overhead. Most of the 1080p difference is inside the GPU dispatch region. Workspace preparation, output transitions and the matching output copy do not explain it. The exact kernel-level cause remains unisolated; no production optimization is justified by these results alone.

## Matched runs

48 fresh processes: 32 shallow partitions/control cases and 16 deeper partitions/control cases. Each cell has two processes, with renderer/model order and instrumentation order reversed in the second round. Each uses 0.75 s warmup and 2 s measurement (at least 30 measured frames). Inputs, model DLL 310.9.16, eight-block schedule, one pass, native kernel bundle and output-copy boundary are matched to the current benchmark fixture. No games, deployed addons, model weights or production kernels were changed.

Headline times are the arithmetic mean of two process GPU medians. Positive delta means Phoenix DLSS5 is slower. Upload/readback and CPU recording are outside the GPU timer. The interval includes queue/dispatch gaps and barriers; it is not SM-active time.

| Resolution / measurement | Model | Phoenix DLSS5 ms | NVIDIA DLSS5 ms | Delta microseconds |
| --- | --- | ---: | ---: | ---: |
| 1280x720 / partitioned | Original | 2.525072 | 2.716880 | -191.808 |
| 1280x720 / partitioned | Lightweight FP8 | 1.504960 | 1.511040 | -6.080 |
| 1920x1080 / uninstrumented | Original | 3.869616 | 3.818384 | +51.232 |
| 1920x1080 / uninstrumented | Lightweight FP8 | 2.279536 | 2.225216 | +54.320 |
| 3840x2160 / partitioned | Original | 13.272032 | 12.604848 | +667.184 |
| 3840x2160 / partitioned | Lightweight FP8 | 7.503040 | 7.238000 | +265.040 |

The separate deep-study uninstrumented controls reproduced gaps of 54.816 and 55.840 microseconds. At 720p Phoenix DLSS5 is faster; at 4K the gap is much larger and model-dependent. That rules out a resolution-independent 60-microsecond tax.

## Where the 1080p time goes

Deeper partition: Phoenix DLSS5 minus NVIDIA DLSS5, in microseconds. Each entry compares the mean of two process stage medians, not the stage means.

| GPU interval | Original | Lightweight FP8 |
| --- | ---: | ---: |
| Engine preparation | +1.088 | +0.640 |
| Dispatch region | +40.560 | +51.792 |
| Internal output transition | +0.000 | +0.000 |
| Transition to copy | +0.000 | +0.000 |
| Matching output copy | +2.976 | +2.880 |
| Restore transition | +0.016 | +0.000 |

Phoenix preparation includes its counter-prefix clear and ordering, ending immediately before the first neural dispatch. Its dispatch region ends at the final group marker, before the output state transition. NVIDIA preparation ends at the first observed NVAPI launch; its dispatch region includes the remaining Evaluate recording. These are useful bounding intervals, not proof that vendor internals have identical phases.

The native extra output transition measures about 0.064 microseconds in absolute time, not 50-60 microseconds. Native launch counts remain 154 / 78 versus 156 / 80 observed vendor launches for Original / Lightweight: extra Phoenix neural launches are not the explanation. Launch counts do not imply equal instructions or memory traffic.

## Controls and limitations

- The shallow three-extra-timestamp instrumentation changed 1080p total time by -0.384 to +0.928 microseconds relative to its controls. The deeper five-extra-timestamp instrumentation changed it by -4.624 to +5.440 microseconds. Original deep profiling therefore narrows the observed gap by about 10 microseconds; its stage attribution must not be presented as the exact uninstrumented gap.
- Empty five-timestamp command lists measured about 1.984-2.016 microseconds; seven-timestamp lists measured about 2.304-2.336 microseconds. These totals include command-stream/boundary costs, not isolated timestamp instruction latency. Do not subtract them from model timings as a correction.
- Every sampled frame passed monotonic timestamp and exact integer-tick partition conservation checks. Aggregated medians are not additive, even though each frame sums exactly.
- Means and medians are stored separately. NVIDIA Lightweight in the shallow 1080p run had a timing tail: comparing stage means with a median headline would give a misleading attribution. Keep the same statistic on both sides.
- All 48 cases passed finite/reset-repeatability checks and exact native/vendor capture equality within each model/extent, including instrumentation controls. Original and Lightweight are not compared for image equality.
- Dynamic clocks, cache state and scheduling remain sources of variation. Two processes per cell support reproduction, not a confidence interval or cross-GPU guarantee.

The matching 1080p deltas could reflect shared work, but that remains a hypothesis. Per-kernel timestamps or a GPU trace with matched launch boundaries are needed to distinguish kernel execution, memory traffic and scheduling. There is no evidence here for a fixed UI/configuration cost or a hidden extra model pass.

## Reproduction and provenance

Run `python tools/investigate_gpu_gap.py`, then the same command with `--deep`. These require the external model, SDK and verified local native bundle described by the runner. `python tools/report_gpu_gap.py` verifies saved case hashes and regenerates this report without GPU work. Temporary probe builds were removed automatically.

[Compact data and measured binary/source identities](gpu-gap.json). Raw local receipts retain samples; the public aggregate contains no machine paths, model binaries or logs. Recorded probe source hashes describe the executed builds, not subsequent documentation or diagnostic-message edits. This follow-up does not replace the broader [current benchmark dataset](README.md).
