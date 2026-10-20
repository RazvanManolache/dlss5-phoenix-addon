# Boundary iteration refactor validation

The same 41-module CUDA source catalogue was built before and after the
structure refactor with CUDA 13.1, sm_120, O3 and the existing per-module
PTX/assembler policies. These are Phoenix DLSS5 native-runtime results,
not a new NVIDIA DLSS5 comparison or an installed-game benchmark.

Baseline: Scheduling structure pass.

GPU/driver/starting state: `NVIDIA GeForce RTX 5080, 616.64, 2315 MiB, 31, 240 MHz, 11.77 W, 0 %`.

## Numerical checks

- All tested RGBA16F reset/history output hashes match the pre-refactor kernels exactly.
- Both Original and Lightweight FP8 were checked at 320x180, 801x603, 1920x1080 and 3840x2160.
- Each process also passed reset repeatability and finite-output checks; the two smaller sizes used D3D12 debug validation.
- 14 x86/x64/isolated-GPU contract cases passed, including 1,000,132 integer-division cases.
- 36/41 complete compiled instruction dumps are byte-identical, including encoded instructions/control words. Changed modules required the paired output/timing checks.
- Frame tests distinguish reflect-once padding from periodic mirroring and staged FP16 conversion from one final conversion.
- Small-model graph/geometry fingerprints and packed-resource rejection cases passed.
- 18,600 GPU scheduling cases cover four tile layouts, partial warps and absent block leaders.
- Signed-word/address checks per architecture: 32-bit: 1,200,041, 64-bit: 1,200,041.
- 1,000,144 full-range unsigned GPU division cases passed, including zero and high-bit divisors.
- 2,017,474 boundary trip-count checks passed for signed/unsigned bounds, wrap, zero stride and production tile extents.

## GPU time

Median of three fresh-process medians per 1080p/4K row, alternating run
order. The scope includes complete engine recording, barriers and matching
output copy, excluding uploads/readbacks and CPU gaps. Smaller sizes had
one diagnostic pair and are omitted from this performance table.

| Phoenix DLSS5 model | Resolution | Before | After | Change |
| --- | --- | ---: | ---: | ---: |
| Original | 1920x1080 | 3.8405 ms | 3.8426 ms | +0.054% |
| Lightweight FP8 | 1920x1080 | 2.2744 ms | 2.2753 ms | +0.039% |
| Original | 3840x2160 | 13.1869 ms | 13.1809 ms | -0.046% |
| Lightweight FP8 | 3840x2160 | 7.4938 ms | 7.4666 ms | -0.363% |

## Compiled resources

Counts come from the compiled modules, not runtime allocation estimates.
Values show before / after; these are not achieved occupancy measurements.

| Modified module | Registers/thread | Static shared bytes | Stack bytes | Local bytes |
| --- | ---: | ---: | ---: | ---: |
| input preparation | 128 / 128 | 4096 / 4096 | 64 / 64 | 0 / 0 |
| window attention channels 32 downsample | 128 / 128 | 0 / 0 | 32 / 32 | 0 / 0 |
| window attention channels 64 downsample | 158 / 158 | 6144 / 6144 | 0 / 0 | 0 / 0 |
| window attention channels 128 downsample | 156 / 156 | 10240 / 10240 | 0 / 0 | 0 / 0 |
| window attention channels 256 downsample | 178 / 178 | 18432 / 18432 | 0 / 0 | 0 / 0 |

No meaningful regression was observed in this bounded run. Small changes
are within normal run/clock variation; they are not claimed speedups.
This does not prove parity or unchanged performance for every GPU, driver,
input, transport or schedule. The fixture was deterministic gradient/detail
RGB with fixed seed and zero motion, not an interactive-game workload.

No games were launched, installed files changed, models copied or releases
published. Temporary before/after bundles are disposable; the receipts and
source/compile recipes are the retained evidence.

See [the structure review](../../KERNEL_STRUCTURE_REVIEW.md) for what remains
instruction-oriented, and [the compact receipt](kernel-boundary.json) for
output/machine-code hashes. Historical reports were not rewritten.
