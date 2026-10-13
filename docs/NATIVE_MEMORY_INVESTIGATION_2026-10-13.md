# Native memory investigation and automatic kernel selection

**Some of the large-image gap was internal memory access and kernel residency,
not transporting images to the GPU.** Controlled tests identified useful changes;
the production native runner now selects two compiled entries according to the
actual neural input size. The model, precision, layer count and launch geometry
are unchanged.

## Automatic selection

- Three 64-channel stages use aligned 16-byte feature/weight reads instead of
  four scalar reads. Their cache modes, asynchronous copies and publication
  fences are preserved. The selected register budgets introduce no spills.
- Global attention and the 512-channel attention projection retain their
  original entries below **18,662,400 input pixels**. At or above that area
  (5760×3240), they use separately compiled 128-register entries.
- This is an area threshold, not a list of allowed resolutions. Portrait,
  ultrawide and odd sizes use the same rule. Selection uses the image fed into
  the network, not the display/output size, and is reevaluated when the network
  is recreated for a changed input size.
- Both entries occupy the same existing modules: **41 modules and 41 created
  function handles**, with no extra installation files. Total kernel code grows
  by **82,024 bytes**. Owned model/frame-buffer allocations are identical in the
  comparisons; this does not measure driver-owned instruction residency.

The threshold is conservative and measured on RTX 5080. It is not a claim that
every possible aspect ratio or another GPU has its optimum at precisely this area.

## Whole-network verification of the automatic path

RTX 5080 16 GiB, driver 616.64, CUDA 13.1; Ryzen 9950X3D, 32 threads, 96 GiB RAM.
Same synthetic RGBA16F color/RG16F zero-motion input and external model DLL
310.8.0.0; full eight blocks, one pass. GPU intervals exclude input uploads and
output readback. Stock clocks were observed, not locked.

Except for the 720p sanity check, standard extents use four fresh processes in
original/automatic/automatic/original order, 256 frames per process, discarding
128. Values average process medians. These are inference measurements, not game FPS.

| Input | Original GPU ms | Automatic GPU ms | Time reduction |
| --- | ---: | ---: | ---: |
| 1280×720 | 2.5074 | 2.5068 | 0.02% — effectively unchanged |
| 1920×1080 | 3.8437 | 3.8180 | 0.67% |
| 3840×2160 | 13.1809 | 13.0899 | 0.69% |
| 5760×3240 | 31.4357 | 30.9592 | 1.52% |
| 7680×4320 | 58.5588 | 57.4222 | 1.94% |

The 720p check uses one process per path and establishes no meaningful speed gain.
The small gains at 1080p/4K remain sensitive to clocks and scheduling. A separate
balanced combination trial measured 2.04% at 6K and 2.16% at 8K, supporting the
large-input benefit without promising a fixed percentage.

The balanced three-route 8K trial measured original Phoenix **58.8259 ms**,
the combined candidate **57.5559 ms**, and NVIDIA **56.9337 ms**. The candidate
reduced that trial's gap from about **3.32% to 1.09%**; NVIDIA still led. Do not
mix NVIDIA's time from that trial with the later automatic-path table.

## What the controlled changes establish

Actual benchmark cubins were inspected with `cuobjdump`; theoretical block
residency was queried through the CUDA driver using each original block size.
These are resource limits, not measured achieved occupancy.

| Stage | Original registers / resident blocks per SM | Selected registers / blocks | Selected spilling |
| --- | --- | --- | --- |
| Global attention, large entry | 176 / 2 | 128 / 4 | None |
| 512-channel projection, large entry | 142 / 1 | 128 / 2 | None |
| 64-channel input view | 128 / 8 | 160 / 6 | None; original had local loads/stores |
| 64-channel output view | 160 / 6 | 160 / 6 | None on either version |
| 64-channel downsample | 123 / 8 | 158 / 6 | None; original had local loads/stores |

- Wider access reduces static global-load instruction counts from **401 to 125**
  for the input view, **401 to 113** for the output view, and **433 to 121** for
  downsampling. This reduces instructions/requests, not the model's data size.
  The three kernel families improved roughly **13–19%** in the diagnostic 8K tests.
- Global attention and the 512-channel projection already had **zero spills**.
  Their large-input improvements therefore cannot be explained as eliminating
  spilling. The lower budgets allow more resident blocks and also change the
  compiler's allocation/scheduling choices.
- The 64-channel input tensor grows from about **16.9 MiB at 1080p** to
  **257.1 MiB at 8K**, compared with a **64 MiB L2 cache**. That supports
  investigating data access, but does not prove a specific cache-miss rate.
- Bypassing L1 made the 64-channel stages over **four times slower**, and the
  512-channel projection about **20% slower**. Their current cache modes matter;
  changing everything to the same mode is harmful.
- Tighter limits are not universally better. Q/K/V projection candidates at
  168 and 192 registers lost about **7–9%** at 8K and were rejected.

## Why the switch is necessary

In diagnostic 1080p timing, the 128-register global-attention entry took about
**0.162 versus 0.097 ms** across the family; the corresponding 512-channel
projection took **0.102 versus 0.073 ms**. Applying the large entries everywhere
gave back the small-resolution memory gains in the whole-network comparison.

The full combined candidate also lost about 1.4% in its initial 4K trial.
We therefore keep the original register choices at 4K, rather than extrapolating
the 8K result downward. The first memory-only 4K trial was noisy and regressed
about 1.0%; its repeat improved 0.78%, and the actual automatic path improved
0.69%. All trials, including these unfavorable results, are retained in the data.

Per-launch timestamps affect scheduling. Kernel-family gains are diagnostic
and must not be added together as a promised full-frame gain. No hardware cache
miss, bandwidth, dynamic spill-traffic or instruction-throughput counters were
available, so the remaining gap is not fully explained.

## Validation and retained alternatives

- **31 alternatives** across six kernel families; 37 experimental compilations
  including six controls. **19,488 GPU frames** executed across screening,
  follow-ups, full-network comparisons and regression validation.
- All candidate reset/final readbacks checked against the original same-size
  fixture match exactly. The NVIDIA comparison checks final output hashes.
  Not every intermediate frame was read back; this is not moving-scene certification.
- Regression validation passes at **eight extents**, including 6K, 8K,
  **5760×3239** immediately below the policy boundary and **5761×3241** above it.
  The actual selected entry names are checked for both branches.
- All **38 preserved original entry instruction bodies** are unchanged.
  The two large-input entry bodies match the tested isolated candidates.
- The model-free addon builds and passes **11 CPU suites and two software D3D12
  suites**. Python tooling tests also pass. Packaging checks that both branches
  were validated and that their entry/threshold metadata has not changed.
- Experimental sources, PTX, cubins, compiler recipes and raw receipts remain
  under `build/native_memory_investigation_20261013/`. Generated fixture files
  and intermediate compiler objects can be recreated and are cleaned afterward.

[Measurements, resources, timing observations and receipt hashes](benchmarks/2026-10-13/native-memory-investigation.json)

The built addon contains the existing portable runtimes plus these native
kernels. It contains neither the NVIDIA model DLL nor extracted model weights.
The public release charts remain the earlier release snapshot; the table above
is the measured source-build follow-up.
