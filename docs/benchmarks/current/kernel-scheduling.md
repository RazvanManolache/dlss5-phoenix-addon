# Kernel scheduling follow-up validation

The same 41-module CUDA source catalogue was built before and after the
structure refactor with CUDA 13.1, sm_120, O3 and the existing per-module
PTX/assembler policies. These are Phoenix DLSS5 native-runtime results,
not a new NVIDIA DLSS5 comparison or an installed-game benchmark.

Baseline: First structure pass.

GPU/driver/starting state: `NVIDIA GeForce RTX 5080, 616.64, 2439 MiB, 31, 577 MHz, 13.04 W, 0 %`.

## Numerical checks

- All tested RGBA16F reset/history output hashes match the pre-refactor kernels exactly.
- Both Original and Lightweight FP8 were checked at 320x180, 801x603, 1920x1080 and 3840x2160.
- Each process also passed reset repeatability and finite-output checks; the two smaller sizes used D3D12 debug validation.
- 13 x86/x64/isolated-GPU contract cases passed, including 1,000,132 integer-division cases.
- 34/41 complete compiled instruction dumps are byte-identical, including encoded instructions/control words. Changed modules required the paired output/timing checks.
- Frame tests distinguish reflect-once padding from periodic mirroring and staged FP16 conversion from one final conversion.
- Small-model graph/geometry fingerprints and packed-resource rejection cases passed.
- 18,600 GPU scheduling cases cover four tile layouts, partial warps and absent block leaders.
- Signed-word/address checks per architecture: 32-bit: 1,200,041, 64-bit: 1,200,041.

## GPU time

Median of three fresh-process medians per 1080p/4K row, alternating run
order. The scope includes complete engine recording, barriers and matching
output copy, excluding uploads/readbacks and CPU gaps. Smaller sizes had
one diagnostic pair and are omitted from this performance table.

| Phoenix DLSS5 model | Resolution | Before | After | Change |
| --- | --- | ---: | ---: | ---: |
| Original | 1920x1080 | 3.8493 ms | 3.8433 ms | -0.156% |
| Lightweight FP8 | 1920x1080 | 2.2755 ms | 2.2751 ms | -0.017% |
| Original | 3840x2160 | 13.1868 ms | 13.1791 ms | -0.058% |
| Lightweight FP8 | 3840x2160 | 7.5044 ms | 7.4842 ms | -0.269% |

## Compiled resources

CUDA Driver occupancy queries use zero dynamic shared memory and launch no
kernels. Maximum active blocks/SM is a resource limit, not measured achieved
occupancy. Binary hashes bind these queries to the compared modules.

| Module | Registers/thread before / after | Max blocks/SM before / after |
| --- | ---: | ---: |
| decoder input upsampling | 236 / 236 | 4 / 4 |
| feed forward contract | 168 / 168 | 3 / 3 |
| feed forward expand | 144 / 144 | 3 / 3 |
| feed forward expand publish | 148 / 148 | 3 / 3 |
| projection | 130 / 130 | 3 / 3 |
| projection wait | 142 / 142 | 3 / 3 |
| query key value | 167 / 168 | 3 / 3 |

All queried modules retain their static shared-memory footprint.
Local and stack storage remain zero for every queried module.

No meaningful regression was observed in this bounded run. Small changes
are within normal run/clock variation; they are not claimed speedups.
This does not prove parity or unchanged performance for every GPU, driver,
input, transport or schedule. The fixture was deterministic gradient/detail
RGB with fixed seed and zero motion, not an interactive-game workload.

No games were launched, installed files changed, models copied or releases
published. Temporary before/after bundles are disposable; the receipts and
source/compile recipes are the retained evidence.

See [the structure review](../../KERNEL_STRUCTURE_REVIEW.md) for what remains
instruction-oriented, and [the compact receipt](kernel-scheduling.json) for
output/machine-code hashes. Historical reports were not rewritten.
