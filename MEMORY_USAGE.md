# Memory ownership audit, 11 October 2026

The native game adapter now shares one temporary workspace and one immutable model across its serial refinement passes. Each pass retains its own two history images, descriptors and sampling state. Both GPU-upload and conventional staged initialization use one model allocation; the staged route also shares one upload buffer and one submission fence.

Test hardware: NVIDIA RTX 5080, driver 616.64; Ryzen 9 9950X3D, 16 cores / 32 threads; 96 GiB installed RAM. Native tables use 1920×1080 RGBA16F color and RG16F motion. HDR format choices and comparison modes can change bridge memory.

| Native refinement passes | Process GPU local before | After | Owned resources after | Owned saving |
| --- | ---: | ---: | ---: | ---: |
| 1 | 676.1 MiB | 676.1 MiB | 493.9 MiB | 0.0 MiB |
| 2 | 1056.2 MiB | 712.2 MiB | 527.7 MiB | 344.1 MiB |
| 4 | 1817.5 MiB | 785.3 MiB | 595.2 MiB | 1032.2 MiB |

Process GPU-local values are DXGI residency snapshots of the whole offline adapter process, including its driver and harness. Owned bytes use D3D12 allocation-info sizes and final resource release callbacks. These columns are different measurements and must not be added. The saving column compares allocations of the same weights/workspace before and after sharing. Native and portable process harnesses also differ.

Native steady ownership at one pass: workspace 203.125 MiB, weights 140.9375 MiB, history 33.75 MiB, bridge images 115.8125 MiB, exposure images 0.25 MiB, exposure readback allocation 0.0625 MiB. Additional passes now add 33.75 MiB of history each rather than repeating the entire workspace/model. Capture readbacks temporarily add 39.6875 MiB for this input format and release after retirement. Conventional staging temporarily adds one 140.9375 MiB upload allocation, independent of pass count.

The previous standalone ledger omitted the direct game runner's bridge images, capture readbacks, guide allocations and external transport resources. The addon now tracks these too. Module/function counters aggregate every refinement pass rather than replacing the preceding pass's counters. Descriptor/pipeline/fence objects are counted, but driver backing byte sizes for those objects remain unknown. Engine leases, timing allocations, OS/driver caches and unlisted CPU container overhead are outside the owned resource total.

Portable host buffers

The noise cache previously stored half-rounded values as floats. It now stores their exact FP16 bits, reducing the 1080p padded cache from 23.90625 to 11.953125 MiB. A 63.75 MiB packed-feature vector is also freed immediately after synchronous upload, before execution, head download and resolve. Backend upload already consumes the source before returning on every installed route; model weights and arithmetic precision are unchanged.

| CPU 1080p worker memory | Before | After |
| --- | ---: | ---: |
| Private commit | 531.6 MiB | 519.4 MiB |
| Resident working set | 579.3 MiB | 567.8 MiB |
| Peak resident working set | 750.0 MiB | 674.8 MiB |

These are one matched 1080p CPU inference process per version, using the same broker/backend and saved input. They are observed process values rather than a statistical memory benchmark. The roughly 75 MiB peak reduction agrees with the lifetimes removed; allocator retention and driver/OS state affect process snapshots. Private commit is not resident physical RAM. The caller's IPC/input/output planes are in another process and are excluded from this worker table.

| Portable route / hardware | Requested backend buffers | Worker resident RAM | Noise cache |
| --- | ---: | ---: | ---: |
| cpu | 467.0 MiB | 567.8 MiB | 11.95 MiB |
| cuda | 467.0 MiB | 438.9 MiB | 11.95 MiB |
| hip_amd | 467.0 MiB | 138.8 MiB | 11.95 MiB |
| sycl_cpu | 467.0 MiB | 831.7 MiB | 11.95 MiB |
| vulkan_amd | 467.0 MiB | 655.5 MiB | 11.95 MiB |
| vulkan_nvidia | 467.0 MiB | 274.0 MiB | 11.95 MiB |

CPU and SYCL CPU use the Ryzen CPU; CUDA and Vulkan NVIDIA use RTX 5080; HIP AMD and Vulkan AMD use the integrated AMD Radeon. Requested backend buffer bytes describe graph allocations, not physical GPU residency. They exclude shared host noise/history, temporary preparation vectors, transport and driver caches. Worker resident RAM includes runtime and mapped/shared pages. New status fields expose noise/history vector capacities and a sampled temporary-capacity high-water mark without pretending to measure the entire allocator peak.

Validation

- Seven direct fixtures: four production sizes plus dark, HDR and motion patterns. All 336 output frames per route match the preserved implementation exactly, including the forced-staging route; each fixture recreates 1→2→4→1 passes. All capture readbacks retire; every owned generation reaches zero live allocations and all kernel modules/functions retire. Failed construction is checked too.
- A blocked-queue upload test confirms staging remains alive before fence completion and retires afterwards.
- All 42 saved portable frames pass on six installed routes at three sizes; nine additional packed-feature checks match across three extents and three seeds. CPU preflight/history/cancellation contracts pass.
- GPU→worker→GPU offload passes temporal-history, motion, depth rejection, stale-generation, expiry and alpha checks. Pixel error is 5.96e-08; all tracked transport allocations retire after shutdown. Its small 96×64 fixture is a lifecycle check, not a 1080p memory/performance result.
- Four counterbalanced ABBA/BAAB blocks (16 isolated processes, 128 frames each) measure native GPU medians of 3.848152 ms before and 3.849160 ms after. The +0.026% difference is small; this pass establishes no material speed change. These intervals are inference measurements, not game FPS. Portable timings collected during correctness checks are not claimed as comparative benchmarks.
- Eleven C++ CPU suites, one software-D3D12 suite, 28 Python tests and all 118 embedded files pass. No model or NVIDIA DLL is embedded. The final addon is built, but no game installation or live-game memory validation was performed.

Source backup: `backup/memory-gap-audit-20261011`. Fixes are committed separately for direct ownership/sharing, portable buffers, transport accounting and staged sharing. Historical source folders and alternatives were retained.

Historical local addon: `build/memory_gap_final_embedded/phoenix.addon64`. SHA256 `7e50aab59fc2cd18ce59a5a7bf47a570bab337ce8da8cbe376e9732d45d65474`.

Historical local evidence directories under `build/`: `memory_gap_shared_staging_validation`, `memory_gap_portable_validation` and `memory_gap_offload_validation_v2`. Raw local receipts are not release contents.

## Optional resolution bridge and temporal upscalers

Normal mode retains the previous allocation path. Reduced-resolution NR uses
a smaller model workspace and history, plus resampling scratch and optional
scaled guides. DLSS SR or XeSS adds a full-output RGBA16F texture (about
15.8 MiB of texels at 1920×1080, with allocator alignment measured separately)
and vendor-private storage. Owned output/guide/compositor allocations are
tracked; NVIDIA/Intel internal resources are **not** included in that ledger.
Process GPU-local usage includes vendor and harness resources and is not an
internal-memory breakdown. Native Phoenix plus either temporal upscaler passed
1080p execution and zero-owned-object teardown checks; these were regression
checks, not a new controlled performance/memory comparison against Normal.
