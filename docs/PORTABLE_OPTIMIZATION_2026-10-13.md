# Portable runner architecture experiments — 13 October 2026

This is a validated local candidate, not a claim about the published release or
in-game performance. The native RTX runner is unchanged. These changes improve
the alternate CUDA and Vulkan runners, which remain substantially slower.

Hardware: RTX 5080 16 GiB, driver 616.64; Ryzen 9 9950X3D, 16 cores / 32 threads;
96 GiB RAM; CUDA 13.1 targeting sm_120. External model: DLSS NR 310.8.0.0.
The [machine-readable measurements](benchmarks/2026-10-13/portable-architecture.json)
include model and addon hashes, sample ranges and memory definitions.

## Whole-frame measurements

1920×1080 RGB32F host/broker input, fractional motion and persistent history,
one sampling pass, all eight transformer repeat counts set to one. Three fresh
processes per variant, twelve frames each, first two discarded. The table uses
the median of the three process medians. The ordering rotates between rounds.
Model time is the GPU operator interval; full call also includes host work,
transport and synchronization. Native RTX and NVIDIA DLSS5 DLL timings use a different
entry path and are not substituted into this paired comparison.

| Runner / configuration | Model ms | Full call ms | Owned GPU MiB |
| --- | ---: | ---: | ---: |
| CUDA baseline 085, 512 windows | 65.71 | 110.66 | 453.38 |
| CUDA candidate 106, 512 windows — default | 56.09 | 99.08 | 453.38 |
| CUDA candidate 106, 1,024 windows | 54.77 | 99.77 | 475.38 |
| Vulkan NVIDIA DLSS5 baseline 085, 512 windows | 111.97 | 162.60 | 453.38 |
| Vulkan NVIDIA DLSS5 candidate 106, 512 windows — default | 101.39 | 150.38 | 453.38 |
| Vulkan NVIDIA DLSS5 candidate 106, 1,024 windows | 97.80 | 150.54 | 475.38 |

The default reduces model time by 14.6% for CUDA and 9.4% for Vulkan. Larger
batches reduce GPU model time further but do not reliably improve the complete
call, so the default stays at 512. The allowed experimental range is 16–8,192.
The earlier 2,048- and 4,096-window sweeps were slower than the 1,024 choice and
used progressively more memory.

Owned GPU allocations count the graph's weights and scratch, excluding driver
caches and other applications. At the default, weights use 144.51 MiB and scratch
308.88 MiB. Scratch allocation size is unchanged: block fusion avoids traffic
through some intermediates but does not yet remove their graph allocations.

Worker RAM also has a cost. Median working sets were 302.4 MiB for CUDA and
270.3 MiB for Vulkan. Median private committed bytes were 909.0 and 740.8 MiB,
respectively; these are not physical-RAM or VRAM measurements. Vulkan's private
commit increased from the baseline's 689.8 MiB despite nearly unchanged working
set and owned GPU buffers. No reduction in total driver memory is claimed.

## Individual operations

Real first-frame operands, preserved against scratch reuse; 32 repetitions,
two warmups and three GPU samples per process. Two fresh processes per variant,
with reversed order. Prefix, transfers, snapshot and graph capture are excluded.
These are operation/block timings, not frame times.

| Piece | CUDA baseline → candidate ms | Vulkan baseline → candidate ms |
| --- | ---: | ---: |
| 128-channel expansion, 32,640 rows | 0.15107 → 0.08785 | 0.25465 → 0.25913 |
| Global expansion, 640×4,096×1,024 | 0.10408 → 0.06877 | 0.30191 → 0.29964 |
| Global normalization, 640 keys / 32 heads | 0.23280 → 0.23246 | 0.87195 → 0.22800 |
| 32-channel feed-forward pair | 0.03190 → 0.02391 | 0.04381 → 0.04354 |
| Complete 32-channel window block | 0.09698 → 0.05930 | 0.14319 → 0.13083 |

The complete JSON retains unchanged and slower pieces too. Small differences
are not treated as proven improvements. See [the piece benchmark instructions](PORTABLE_PIECE_BENCHMARKS.md)
to reproduce the method with external runtime/model/input files.

## Selected changes

- Batch the independent per-head middle projections into one strided GEMM.
  This removes launches without transposing activations or copying weights.
- Specialize CUDA publication and choose output tile widths for measured matrix
  families. A wider tile helps QKV and some global expansions but hurts several
  contractions; it is not applied uniformly. Automatic tile routing applies
  only to compute capability 12.0, where it was measured; other CUDA devices
  retain their existing column choices.
- Keep the eligible 32-channel window block's nine operations in one CUDA CTA.
  Intermediate activations, Q/K/V and attention scratch stay in shared memory.
  The original gates, rounding boundaries and residual operations remain.
- Initialize Vulkan's parallel row pipeline. Previously its shader was shipped
  but never initialized, leaving the provider on its scalar fallback. Specialize
  the bounded positive half arithmetic in global normalization while retaining
  its original reduction order.

The first generic whole-window fusion was slower. Specializing known dimensions,
formats and residual type reduced registers from 128 to 72–75 per thread and
made it faster. Both versions had zero stack/local spills. The selected kernel
uses 48 KiB of explicit shared arrays; compiler resource output reports 49 KiB.

## Alternatives retained

- The 64-channel shared feed-forward version was slower in the measured piece.
  It remains opt-in with `cuda_fused_wide_feed_forward`; default is false.
- `cuda_fused_window_block`, `cuda_fused_feed_forward` and `cuda_shape_matrices`
  select the winning CUDA changes. They default to true in the runtime; an
  explicit `cuda_fp8_columns` choice disables automatic matrix selection unless
  `cuda_shape_matrices` is explicitly enabled.
- cuBLASLt matched all ten synthetic dyadic matrix hashes. It helped some large
  projections and lost on others. Its benchmark is retained as an optional
  target; neither the test executable nor cuBLASLt is embedded in the addon.
- A tensor shortcut for large global QK and probability×V matrices was fast in
  isolation but changed output. The probability×V path caused the regression:
  eight-frame channel MAE reached about 0.0052 and maximum error reached 0.0952.
  QK alone passed 32 temporal A/B frames, but the whole-model improvement was
  small and inconsistent. Neither shortcut replaces the selected path.

Local alternatives, sources, jobs, binary hashes and raw reports are retained
under `build/portable_arch088` and the numbered `portable_arch*` build/runtime
directories. Compiler object and linker intermediate files are regeneratable.

## Validation and remaining limits

- 216 exact temporal A/B frames at 1080p for the final paired measurements.
- 42 saved-reference frames at 1080p, 321×193 and 96×64, covering CUDA RTX,
  Vulkan RTX, CPU, HIP on the available AMD iGPU, SYCL CPU and Vulkan AMD iGPU.
  This does not validate an Intel GPU or a discrete AMD GPU.
- Six synthetic window/residual combinations, three replays each, match nine
  separate CUDA operations exactly. CUDA, Vulkan RTX and Vulkan AMD operator
  contracts cover layout tails, batch strides, normalization, feature packing
  and 1,354,743 finite conversion checks per route.
- The candidate addon passes its build, CPU and software D3D12 suites and embeds
  120 model-free runtime/native files. No game installation or live game test
  was performed during this pass.

The remaining gap is architectural: many window chunks, repeated half/FP8
conversion and intermediate traffic, plus host history/resolve and transport in
the full call. Fusion helps, but the portable graph still differs substantially
from the native kernels. Faster unordered reductions cannot be accepted merely
because their isolated timings look good.
