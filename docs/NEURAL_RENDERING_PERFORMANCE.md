# Neural rendering performance evaluation

The [current two-model benchmark](benchmarks/current/README.md) supersedes the older headline model/backend timings below with fresh native RTX/NVIDIA, CUDA, Vulkan NVIDIA/AMD and CPU data. This dated report retains its original advanced-adapter and pre-fix measurements and is not rewritten as a current run.

> Correctness update (2026-10-18): the native graph packaging/layout failures at 1024x768 and 4K are fixed in the experimental candidate. See [repair evidence](../artifacts/current/native-correctness/REPORT.md). The timings below remain the recorded pre-fix measurements.

2026-10-18. Measured on NVIDIA RTX 5080 (16 GB, driver 616.64), Ryzen 9 9950X3D. The packaged candidate and unchanged SDK were used; runtime binaries were not replaced during the tests.

The sweep attempted 25 standard resolutions from 320x180 through 7680x4320, including 4:3, 16:10 and ultrawide modes. All eight central blocks ran once, with one full network pass. Input, inference and output dimensions matched. Native calls kept textures on the GPU; portable calls used host float planes. Synthetic structured pixels, zero motion and accepted temporal continuation were used.

## Native costs

Median complete-call milliseconds, including native input preparation and GPU synchronization. Map/UI/field creation and host image transfer were outside these native calls.

| Resolution | Unchanged SDK | Candidate default | Style + map + UI + field |
| --- | ---: | ---: | ---: |
| 1280x720 | 2.86 | 2.85 | 3.09 |
| 1920x1080 | 4.25 | 4.24 | 4.91 |
| 2560x1440 | 6.44 | 6.41 | 7.48 |
| 3440x1440 | 8.62 | 8.58 | 9.97 |
| 3840x2160 ! | 13.84 | 13.91 | 15.85 |
| 5120x1440 * | 12.51 | 12.49 | 26.58-30.78 |
| 5120x2160 * | 18.79 | 18.78 | 21.31-38.25 |
| 5120x2880 * | 25.07 | 25.04 | 28.71 |
| 7680x4320 | 57.21 | 57.24 | 65.60 |

* Selected longer-warmup repeats are listed; 5120-wide results show substantial variation. Ranges are observed run medians, not confidence intervals. ! Output repeatability is not validated at that size.

![Native results](../artifacts/current/neural-performance/native-performance.png)

Native default performance and tracked default memory are close to the old SDK in the stable cases. At 1080p the optional suite adds about 0.67 ms per call; at 1440p about 1.06 ms; at 4K about 1.94 ms. The 5120-wide paths have much larger or inconsistent costs, so a general cheap-overhead claim is not justified.

## Portable costs

At 1080p, the feature-isolation run measured about 82 ms for the default CUDA host API and 97 ms for Vulkan. Combined effects took about 555 and 567 ms. The host advanced API without extra planes already cost about 108/127 ms. Natural style alone cost about 238/253 ms; a map about 159/174 ms; UI about 156/167 ms; distortion about 324/341 ms.

These are different interfaces from native GPU-resident calls. The portable boundary currently performs canonicalization, grading/composition and snapshots on the CPU; those costs dominate the new feature overhead. Moving boundary work to the GPU and removing unnecessary advanced-path work is a clear optimization target. The portable neural backend has its own baseline cost.

![Portable results](../artifacts/current/neural-performance/portable-performance.png)

8K Vulkan failed device-memory allocation in both versions. The CUDA backend alone requests about 16.04 GiB: default calls measured about 8.15 seconds and combined effects about 16.15 seconds, based on only two and one timed calls. Those are memory-pressure observations, not reliable percentile estimates.

## Memory and correctness

Tracked native library allocations increased from about 403 to 563 MiB at 1080p, 1088 to 1693 MiB at 4K, and 3.82 to 6.18 GiB at 8K. These figures exclude caller textures/driver caches. Advanced buffers remain cached for the session after disabling effects.

- Native 1024x768 failed on its first frame in both old and new DLLs, after session creation succeeded.
- Native 3840x1600 showed large reset differences in both versions: mean RGB error around 0.32-0.35 and maximum around 0.86-0.91 in the diagnostic fixture. This is a correctness blocker.
- Native 3840x2160 also showed reset variation in both versions, with maximum differences around 0.015-0.019 in repeated calls.
- 1080p repeats matched exactly. The original sweep found exact reset hashes across all four before/candidate/return/after references at 22 of the 24 completed native sizes.

The output diagnostics are separate from timing: no image-correctness certification follows from successful timings. Numerical details are in repeatability.json.

## Method and limitations

Each ordinary case used at least 0.75 seconds of warm-up and a 2-second measurement target. Slow cases stopped after a 15-second measurement window, so large portable cases have fewer than 12 samples. Flagged native/portable timing repeats used 5-second warm-up and 5-second measurement targets. Medians, p95, raw samples and per-case memory scopes are saved.

Desktop/GPU activity was not locked down and clocks were not forced. NVML briefly listed other DLSS screen-validation entries near the repeats, but they were absent from a follow-up live-process check; overlap and causation cannot be established. The 5120-wide variation should be confirmed under controlled idle-GPU conditions before assigning a regression or improvement.

These are standalone processing measurements, not game FPS. Engine map production, the game render, upscalers, presentation and helper IPC were not included. No game was launched by this evaluation and no NVIDIA model runtime DLL was executed; normal GPU drivers/NVAPI were used.

Full 25-resolution tables, individual 1080p feature costs, failure records, timings and source/binary hashes: [current report](../artifacts/current/neural-performance/REPORT.md), [native CSV](../artifacts/current/neural-performance/native-resolutions.csv), [portable CSV](../artifacts/current/neural-performance/portable-resolutions.csv), [summary JSON](../artifacts/current/neural-performance/summary.json).

Native and portable use different structured patterns/sampling sequences; before/after inputs match within each route. Cross-route timings are not an exact-pixel matched benchmark.
