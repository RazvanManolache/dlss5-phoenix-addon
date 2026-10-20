# Kernel structure refactor validation

The same 41-module CUDA source catalogue was built before and after the
structure refactor with CUDA 13.1, sm_120, O3 and the existing per-module
PTX/assembler policies. These are Phoenix DLSS5 native-runtime results,
not a new NVIDIA DLSS5 comparison or an installed-game benchmark.

GPU/driver/starting state: `NVIDIA GeForce RTX 5080, 616.64, 2497 MiB, 28, 225 MHz, 9.35 W, 0 %`.

## Numerical checks

- All tested RGBA16F reset/history output hashes match the pre-refactor kernels exactly.
- Both Original and Lightweight FP8 were checked at 320x180, 801x603, 1920x1080 and 3840x2160.
- Each process also passed reset repeatability and finite-output checks; the two smaller sizes used D3D12 debug validation.
- 13 x86/x64/isolated-GPU contract cases passed, including 1,000,132 integer-division cases.
- 27/41 complete compiled instruction dumps are byte-identical, including encoded instructions/control words. Changed modules required the paired output/timing checks.
- Frame tests distinguish reflect-once padding from periodic mirroring and staged FP16 conversion from one final conversion.
- Small-model graph/geometry fingerprints and packed-resource rejection cases passed.

## GPU time

Median of three fresh-process medians per 1080p/4K row, alternating run
order. The scope includes complete engine recording, barriers and matching
output copy, excluding uploads/readbacks and CPU gaps. Smaller sizes had
one diagnostic pair and are omitted from this performance table.

| Phoenix DLSS5 model | Resolution | Before | After | Change |
| --- | --- | ---: | ---: | ---: |
| Original | 1920x1080 | 3.8314 ms | 3.8313 ms | -0.002% |
| Lightweight FP8 | 1920x1080 | 2.2625 ms | 2.2626 ms | +0.003% |
| Original | 3840x2160 | 13.1379 ms | 13.0664 ms | -0.544% |
| Lightweight FP8 | 3840x2160 | 7.4818 ms | 7.4925 ms | +0.144% |

No meaningful regression was observed in this bounded run. Small changes
are within normal run/clock variation; they are not claimed speedups.
This does not prove parity or unchanged performance for every GPU, driver,
input, transport or schedule. The fixture was deterministic gradient/detail
RGB with fixed seed and zero motion, not an interactive-game workload.

No games were launched, installed files changed, models copied or releases
published. Temporary before/after bundles are disposable; the receipts and
source/compile recipes are the retained evidence.

See [the structure review](../../KERNEL_STRUCTURE_REVIEW.md) for what remains
instruction-oriented, and [the compact receipt](kernel-structure.json) for
output/machine-code hashes. Historical reports were not rewritten.
