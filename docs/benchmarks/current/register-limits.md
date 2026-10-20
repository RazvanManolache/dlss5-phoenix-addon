# Two-model register and spill recheck

## Main findings

- Both models passed all tested output checks across 12 model/extent pairs and 78 fresh processes.
- Register spilling exists in both models, but this screen found no output corruption caused by the tested budgets.
- Relaxing eight budgets reduces the module inventory from 16 spilling entries to 12, but does not produce a clear end-to-end gain.
- Tighter budgets are roughly 21-39% slower at the repeated 1080p/4K/8K points. Higher theoretical occupancy does not compensate for the extra spills in this mix.
- Small timing differences are not stable optimization evidence: even equivalent entry-control profiles vary, and one 8K profile is especially noisy.
- Production budgets and the large-image selector were left unchanged.

## What the earlier test was

The matching historical experiment is commit `f86c2f2` and the
[13 October investigation](../../NATIVE_MEMORY_INVESTIGATION_2026-10-13.md).
It studied CUDA register pressure, local-memory spills and the large-image
entry switch, not Windows registry overflow. It used the older 310.8 DLL
and Original model. This recheck uses both Original and Lightweight FP8
from 310.9.16, the current refactored kernels and CUDA 13.1 / sm_120.

GPU/driver/starting state: `NVIDIA GeForce RTX 5080, 616.64, 2363 MiB, 32, 862 MHz, 17.02 W, 4 %`.

## What spilling means

A register budget is not a buffer that corrupts neighboring registers.
When live values exceed the available allocation, the compiler can spill
them into thread-local memory. More registers can reduce spilling but also
reduce resident blocks. Neither fewer spills nor higher theoretical
occupancy alone establishes faster inference.

The production build contains 41 modules / 43 entries; 16 entries
have nonzero compiler-reported spill loads or stores. Only selected entries
with nonzero launches in the model plan execute. At the smaller extents,
Original uses 15 spilling entries; Lightweight FP8 uses 12. At/above the
automatic large-image threshold those counts become 16 and 13, respectively.

Compiler spill bytes are static quantities, not measured bytes per frame.
Driver local bytes describe per-thread local storage; maximum blocks/SM
is theoretical resource capacity, not measured achieved occupancy.
`cuobjdump LOCAL:0` alone is not a spill-free verdict: it can coexist
with nonzero STACK and PTXAS spill counts in these binaries.
No dynamic spill-traffic, bandwidth or hardware-stall counters were collected.

## Production spill inventory

| Entry | Registers | Stack bytes | Spill stores / loads | Max blocks/SM |
| --- | ---: | ---: | ---: | ---: |
| global_attention_split_heads_512_feed_forward_input_view_tile_sync | 176 | 8 | 8 / 8 | 2 |
| global_attention_split_heads_512_projection_large_input | 128 | 8 | 4 / 4 | 2 |
| global_attention_split_heads_512_projection_pool | 144 | 40 | 44 / 128 | 3 |
| output_reconstruction | 128 | 104 | 92 / 92 | 16 |
| input_preparation | 128 | 64 | 64 / 64 | 16 |
| window_attention_channels_32_attention | 127 | 48 | 44 / 44 | 16 |
| window_attention_channels_32_downsample | 128 | 32 | 32 / 28 | 16 |
| window_attention_channels_32_input_view | 128 | 72 | 72 / 72 | 16 |
| window_attention_channels_32_output_view | 128 | 64 | 60 / 60 | 16 |
| window_attention_channels_32_upsample | 128 | 40 | 36 / 36 | 16 |
| window_attention_channels_64_attention | 128 | 40 | 40 / 40 | 8 |
| window_attention_channels_128_input_view | 156 | 40 | 40 / 40 | 3 |
| window_attention_channels_128_output_view | 156 | 8 | 8 / 8 | 3 |
| window_attention_channels_128_upsample | 165 | 8 | 8 / 8 | 3 |
| window_attention_channels_256_attention | 184 | 8 | 8 / 8 | 1 |
| window_attention_channels_256_upsample | 128 | 40 | 40 / 40 | 2 |

The large split-window projection currently spills 4 store / 4 load bytes;
the older investigation reported none. This is a newly checked difference
in the current build, not proof that a particular edit caused it.

## Isolated variations

- **Production:** unchanged register choices and automatic input-area selection.
- **Relaxed:** higher budgets on eight selected spill-bearing kernels.
- **Tighter:** lower budgets on the same eight kernels.
- **Default entries:** disable the two large-image entry switches only.
- **Large entries:** force those two large-image entries at every size.

Precision, graph, weights, block sizes, cache modes, assembler optimization
and store/synchronization code are unchanged. Eight unchanged-budget PTX
reassembly controls have identical instruction dumps to production, so the
PTX build route is not an uncontrolled code-generation change.

| Tuned entry | Production registers | Relaxed budget / used | Tighter budget / used |
| --- | ---: | ---: | ---: |
| input_preparation | 128 | 160 / 160 | 96 / 96 |
| output_reconstruction | 128 | 160 / 160 | 96 / 96 |
| window_attention_channels_32_input_view | 128 | 160 / 160 | 96 / 96 |
| window_attention_channels_32_output_view | 128 | 160 / 128 | 96 / 96 |
| window_attention_channels_64_attention | 128 | 160 / 159 | 96 / 96 |
| window_attention_channels_128_input_view | 156 | 192 / 192 | 128 / 128 |
| window_attention_channels_256_upsample | 128 | 160 / 160 | 96 / 96 |
| global_attention_split_heads_512_projection_pool | 144 | 176 / 155 | 112 / 112 |

## Concrete register tradeoffs

The following resource changes are real, but their individual timing effect
is not isolated by the multi-kernel mix:

| Entry | Profile | Registers | Spill stores / loads | Max blocks/SM |
| --- | --- | ---: | ---: | ---: |
| input_preparation | production | 128 | 64 / 64 | 16 |
| input_preparation | relaxed | 160 | 0 / 0 | 12 |
| input_preparation | tighter | 96 | 680 / 680 | 20 |
| global_attention_split_heads_512_projection_pool | production | 144 | 44 / 128 | 3 |
| global_attention_split_heads_512_projection_pool | relaxed | 155 | 0 / 0 | 3 |
| global_attention_split_heads_512_projection_pool | tighter | 112 | 1032 / 1148 | 4 |

## Numerical and GPU checks

Every successful variation must match the production reset A, one-frame
history and reset B output hashes for its model and exact input size.
Finite-output, reset-repeatability and D3D12 error checks also pass before
a row receives a timing. The 801x603 cases enable D3D12 debug validation.

At 1080p, 4K and 8K each profile uses two fresh processes, with the second
profile order reversed. Values are medians of those process medians.
Odd-size and threshold checks use one process/profile: correctness and
policy diagnostics, not reliable fine-grained performance rankings.
Each process warms for at least 0.5 seconds and measures at least 1 second
and 30 frames. GPU timestamps include engine recording/barriers/output copy,
excluding uploads, readbacks and CPU gaps. Stock clocks were not locked.

| Phoenix DLSS5 model | Input | Production ms | Relaxed | Tighter | Default entries | Large entries |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original | 801x603 | 2.1231 | 2.1193 (-0.18%) | 2.3619 (+11.25%) | not tested | not tested |
| Lightweight FP8 | 801x603 | 1.2321 | 1.2269 (-0.42%) | 1.4286 (+15.95%) | not tested | not tested |
| Original | 1920x1080 | 3.8452 | 3.8794 (+0.89%) | 4.6413 (+20.70%) | 3.8515 (+0.16%) | 3.8808 (+0.92%) |
| Lightweight FP8 | 1920x1080 | 2.2776 | 2.3112 (+1.48%) | 3.0386 (+33.41%) | 2.2768 (-0.03%) | 2.3021 (+1.08%) |
| Original | 3840x2160 | 13.2280 | 13.3371 (+0.82%) | 16.2785 (+23.06%) | 13.2082 (-0.15%) | 13.1316 (-0.73%) |
| Lightweight FP8 | 3840x2160 | 7.5069 | 7.6122 (+1.40%) | 10.4200 (+38.81%) | 7.5207 (+0.18%) | 7.5360 (+0.39%) |
| Original | 5760x3239 | 31.4178 | not tested | not tested | 31.3976 (-0.06%) | 30.9675 (-1.43%) |
| Lightweight FP8 | 5760x3239 | 18.1302 | not tested | not tested | 18.1247 (-0.03%) | 17.9386 (-1.06%) |
| Original | 5760x3240 | 30.9533 | not tested | not tested | 31.2946 (+1.10%) | 30.9832 (+0.10%) |
| Lightweight FP8 | 5760x3240 | 17.8904 | not tested | not tested | 18.1074 (+1.21%) | 17.8951 (+0.03%) |
| Original | 7680x4320 | 56.7613 | 57.2345 (+0.83%) | 68.8166 (+21.24%) | 57.5345 (+1.36%) | 56.6839 (-0.14%) |
| Lightweight FP8 | 7680x4320 | 34.6086 | 34.9316 (+0.93%) | 45.6134 (+31.80%) | 36.4662 (+5.37%) | 33.8746 (-2.12%) |

### Timing variability

These repeated-profile process medians differ by more than 3%. This is
a descriptive warning, not a statistical test. The raw samples are retained;
do not treat the corresponding percentage as a stable optimization gain.

| Model / input / profile | Fresh-process medians, ms |
| --- | --- |
| small-fp8 / 7680x4320 / default_entries | 38.4952, 34.4372 |

The automatic threshold is 18,662,400 input pixels: 5760x3239 is
below it, and 5760x3240 reaches it exactly. The receipt includes the selected
entry and launch count for every module in both models. No new model-specific
threshold or blanket register setting was applied to production.

These are bounded synthetic results, not moving-scene/game certification or
universal tuning recommendations. Multi-kernel mixes identify tradeoffs but
do not attribute a timing difference to one kernel. Small differences need
more isolated measurements before any production change.

No games were launched or installed files changed. The external model DLL
was not copied. Temporary experimental bundles were removed; only hashes,
resources and compact results remain. [Machine-readable receipt](register-limits.json).

Reproduce with `tools/benchmark_register_limits.py --baseline <fresh-production-bundle>`,
then `tools/report_register_limits.py`. The production builder intentionally
has no experimental tuning switches.
