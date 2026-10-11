# Reading the native kernels

The production catalogue selects 41 CUDA modules for the 154-operation model
graph. Each module now exports a function named for its operation, such as
`phoenix_image_to_tokens` or `phoenix_window_attention_channels_128_attention`.
The catalogue's `entry` field records that exported name. Its `name` field remains
the graph's compatibility identifier; these two names serve different purposes.

Start with the fully derived layout conversions:

- [Image to tokens](src/native/kernels/production/optimized/layout/image_to_tokens.cu)
- [Tokens to image](src/native/kernels/production/optimized/layout/tokens_to_image.cu)
- [Shared coordinate mapping](src/native/kernels/production/common/feature_layout.cuh)

The tensors on either side of these conversions have shape `[LH, LW, 1024]`
and store E4M3 FP8 features. One 32-bit word carries four feature bytes, giving
256 words per cell. The conversions copy words without decoding or rounding
their contents.

Image storage groups cells into 4-by-4 spatial tiles. Transformer storage groups
16 consecutive cells in raster order. Both use the same internal permutation of
cell lanes and feature words, documented bit by bit in `feature_layout.cuh`.
Image-to-token conversion also zero-fills the sequence to a multiple of 32 cells.
The inverse writes only the real image cells. Model plans align both spatial
dimensions to four; these are not general converters for arbitrary dimensions.

The launch packet has two 64-bit device pointers followed by 32-bit height and
width. Offset and size assertions preserve the existing 24-byte interface.

## Feed-forward reduction

[The reduction helper](src/native/kernels/production/common/feed_forward_reduction.cuh)
expresses one partial-projection accumulation as token-tile and feature-word
coordinates. This temporary reduction storage contains FP16 values, unlike the
FP8 layout-conversion tensors. A 16-token, 1024-channel tile therefore occupies
8192 packed half-pair words.

Each lane contributes four half-pair words to each of four feature groups. The
helper computes a 64-bit byte address and preserves the four vector atomic
updates in their original order. Its call site in
[the contraction kernel](src/native/kernels/production/optimized/global_attention/transformer/feed_forward_contract.cu)
checks each of four token tiles explicitly. The `accumulate_partial_projection`,
`store_final_projection` and `publish_projection_completion` labels identify the
surrounding producer/consumer paths.

Ordinary stores cannot replace these atomics: multiple producer blocks add to
the same reduction storage. The completion flags and memory fences still govern
when another block can consume those contributions. Valid planned offsets are
nonnegative and below the signed 30-bit word-index limit used by this rewrite.

## Boolean expressions and packed arithmetic

Predicate truth-table shifts have been replaced with their derived Boolean
expressions. For example, bit 7 of a three-input truth table means all three
conditions are true. Source masks use eager Boolean `&` and `|` rather than
short-circuit operators, so evaluating a predicate does not add a short-circuit
control-flow dependency. Their operands are already computed Boolean values.

Per-bit identities also replace expanded expressions: `A | (~A & B)` becomes
`A | B`. A seven-flag packet followed by `& 0x40` becomes a selection of the flag at bit
6. Adjacent identical warp votes share one collective result.

[Packed arithmetic helpers](src/native/kernels/production/common/feed_forward_math.cuh)
document the precision boundaries. A half pair contains two FP16 values; its
multiply/add operations round independently per lane. A fused multiply-add
rounds once and must remain fused. FP8 packing saturates finite E4M3 results.
[Conversion helpers](src/native/kernels/production/common/window_math.cuh)
also document half ordering, hardware approximations and warp-wide transposition.

## Validation and remaining work

Build the CUDA modules and addon using [README.md](README.md). Native reference
validation checks reset/history output hashes at four resolutions. The optional
model-free layout comparison additionally checks padding, arbitrary bit patterns,
buffer guards and an unchanged input buffer:

```powershell
python tools/validate_feature_layout.py `
  --baseline C:\backup\baseline_bundle `
  --candidate build\native_release `
  --output build\layout_validation.json
```

This is a partial semantic reconstruction of the implementation. The two layout
conversions and one reduction path now have compact coordinate-based code.
The large attention and projection bodies use structured loops and branches.
Tile iteration and producer waits no longer depend on labels or jumps. The
rewrite was checked by lowering the structured control flow back into a graph
and comparing every operation and conditional edge with the previous graph.
Redundant expression parentheses were removed only when the parsed expression
tree stayed identical; casts, call arguments and required precedence remain.

The control-flow pass removed 1,299 jumps from 39 kernels and 16,871 redundant
parenthesis pairs from CUDA sources and helpers. All 41 modules compiled, and
the four-resolution reset/history reference checks passed. On an RTX 5080, a
counterbalanced 16-run 1920x1080 comparison measured median GPU intervals of
4.333216 ms before and 4.333320 ms after (+0.0024% latency). These measurements
use synthetic regression inputs and exclude host upload/readback; they are not
game frame-rate measurements.

The following readability pass merges adjacent tests of the same predicate while
that predicate stays unchanged. A body that changes the predicate, takes its
address or passes it to a potentially writable reference ends the merge. Local
variable scopes remain where declarations require them.

Single-use address and copy-through temporaries are replaced with direct
expressions only when the integer width stays identical. Boolean `&& true` /
`|| false` terms, full-width unsigned masks and empty statements are removed.
Inline assembly outputs, completion tokens and temporaries used more than once
retain their necessary storage and scopes.

This pass removes 1,805 repeated tests, 2,521 temporary-only scopes, 2,216
redundant full-width masks, 114 Boolean identity terms and 69 empty statements.
The four-resolution reference checks still pass. A counterbalanced 16-run RTX
5080 comparison at 1920x1080 measures 4.333904 ms before and 4.322440 ms after
(-0.2645% latency). Register allocation and spills change in some modules;
source simplification does not guarantee fewer registers or spill instructions.

A final empty-statement scan removed one remaining semicolon after a temporary
compound scope in reconstruction. Required declaration terminators remain.
All 41 rebuilt modules have identical machine instruction/control streams to
the preceding cleaned build, and all four resolution regressions pass.

A fresh NVIDIA DLSS NR comparison at 1920x1080 on the RTX 5080 (driver 616.64)
uses identical synthetic FP16 input/motion and the same model DLL. Eight fresh
processes per runner each execute 128 frames, excluding the first 32. The
median of process medians is 4.322200 ms for Phoenix and 3.859112 ms for NVIDIA:
Phoenix takes 12.00% longer in this workload. Reset and final outputs match
byte-for-byte; intermediate NVIDIA outputs are not captured. Both use D3D12
debug validation, and timing excludes input upload/output readback. This is
a runner comparison, not a game frame-rate measurement.

These bodies still contain reused register-fragment variables. Assigning them
more specific meanings requires deriving their data flow; a global variable-name
substitution would not establish those meanings. CUDA instructions that enforce rounding, cache
behavior or synchronization remain explicit where those details affect results.

## Address, Boolean and synchronization cleanup

The next pass combines 165 scaled-address calculations into named 64-bit address
snapshots. Destination words are still assigned at their original use points;
stable branch predicates allow a snapshot to be shared across two branches.
The launch packet's word-pair representation remains compatible with the runtime.
Nine overflow predicates also feed later control flow and are preserved, with
comments explaining that use. Their liveness is checked through branches, loop
backedges, returns, breaks and continues.

`scaled_word_address` documents an existing overflow quirk: when the low word of
a word offset is at least 2^30, the previous calculation can increment the high
address word beyond ordinary 64-bit addition. The helper preserves this edge
behavior rather than silently changing large offsets. A compiled host copy of
the actual helper matches the previous word arithmetic in 1,000,100 boundary
and deterministic random cases.

354 expanded bitwise networks become equivalent smaller expressions, removing
960 bitwise operations from source. Another 82 masked Boolean packets directly
select the requested flag bits. Truth-table checks establish the bitwise
identities; integer types and widths remain unchanged. A conditional selection
is used at one 128-channel downsampling site because CUDA 13.1 otherwise raises
its spill stores and loads from 48 to 124 bytes despite equivalent arithmetic.

1,888 shuffle instructions no longer produce unused validity outputs. Their
mode, source, lane control, clamp and active-member mask remain unchanged. PTX
permits the predicate destination to be omitted; see NVIDIA's
[shuffle instruction reference](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/shfl_sync.html).

33 consecutive fence sequences collapse to one fence of their strongest scope,
removing 87 fence calls. Fences separated by memory operations or barriers remain.
No block barrier, warp barrier, atomic update or completion publication is removed.
The retained scopes follow NVIDIA's
[memory fence definitions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#memory-fence-functions).

All 41 final modules compile, and the reset/history reference checks pass at
640x384, 1280x720, 1920x1080 and 2560x1080. The checks execute 128 frames per
resolution and compare the reset and final output hashes. Fifteen targeted
arithmetic, liveness, shuffle and fence guard tests pass, alongside the 27
existing Python tests. The source was preserved in a backup archive and the Git
tag `backup/native-before-four-pattern-cleanup-20261011` before edits began.

A final counterbalanced 16-process RTX 5080 comparison (driver 616.64) at
1920x1080 measures 4.326352 ms before and 4.320160 ms after (-0.1431% latency),
using 96 measured frames after 32 warmup frames per process. Performance is
effectively unchanged at this scale. The GPU interval includes preparation,
network and reconstruction, excluding host upload/readback; the input is the
same synthetic fixture used by the reference checks.

Compiler allocation still varies by kernel. The 128-channel upsampling kernel
uses 16 rather than 8 bytes of spill stores/loads; the 256-channel upsampling
kernel drops from 92/116 to 40/40 bytes. Two kernels gain one or two registers,
while five use fewer registers. These are compiler-reported static quantities,
not measured aggregate GPU memory consumption. The selected 128-channel
downsampling form retains its original 48-byte spill stores/loads.

## Native launch scheduling recovery

The large end-to-end regression was in `production_network.h`: it submitted
each group as one NVAPI kernel chain. The recorder now submits individual
launches in their original order, retaining every existing group barrier and
the kernels' tile publication counters. No additional per-launch barrier is
inserted. Four redundant arrays of copied launch descriptors were removed.
The readable CUDA sources, arithmetic and tensor lifetimes are unchanged.

On RTX 5080, driver 616.64, with Ryzen 9 9950X3D (16 cores / 32 threads), a
counterbalanced 1920x1080 comparison measured these complete GPU intervals:

| Runner | GPU evaluation | CPU recording | Record / submit / wait wall |
| --- | ---: | ---: | ---: |
| Phoenix, grouped launches | 4.324288 ms | 0.220100 ms | 4.786775 ms |
| Phoenix, individual launches | 3.846608 ms | 0.279425 ms | 4.350525 ms |
| NVIDIA DLL | 3.874896 ms | 0.386625 ms | 4.489225 ms |

Values are medians of eight isolated process medians per route, with 96
measured frames after 32 warmup frames each. Both Phoenix routes used the
identical newly compiled readable kernels. GPU intervals enclose the complete
runner/Evaluate, excluding upload/readback, with no per-kernel timestamps.
CPU, GPU and wall intervals overlap and must not be added. The 11.05% GPU
latency reduction recovers approximate NVIDIA parity on this synthetic fixture;
the 0.73% difference between the two runners is small. This is not game FPS.

The saved pre-cleanup, already optimized kernels with the previous host launcher measured 4.313432 ms,
versus 4.317608 ms for the readable kernels: about 0.10% difference. That small
change did not explain the large regression. Per-launch attribution pointed to
launch scheduling; the final uninstrumented comparison confirmed it. The exact
driver scheduling mechanism is not established, and other GPUs/drivers need
separate measurements.

All four saved reset/history cases pass for the final source probe, executing
128 frames each with finite outputs and zero D3D12 errors. Separate extent
screens measured 2.114880 to 1.941632 ms at 640x384, 2.865584 to 2.512336 ms at
1280x720, 4.321840 to 3.842464 ms at 1920x1080, and 5.717760 to 4.825280 ms at
2560x1080. These extent screens use one process per route; the 1080p table above
is the repeated comparison. NVIDIA reset and final output hashes also match
byte-for-byte; intermediate NVIDIA frames were not captured.

Submitting each launch with an additional UAV barrier was slower than the
selected form. An arbitrary two-launch subdivision failed fence retirement and
was rejected; its experimental source is preserved in commit `c41aea4`, with
no subdivision option retained in the final probe. Optional `--profile-layers`
is offline attribution only: its timestamps disturb execution and must not be
used to claim end-to-end production performance.

All 41 kernels rebuild, 11 C++ CPU suites, one WARP suite and 27 Python tests
pass, and all 118 payloads in the complete embedded addon verify. No model or
NVIDIA DLL is embedded. The pre-fix source is retained by the tag
`backup/native-before-launch-scheduling-20261011`; the production fix is
`b649062`. No game deployment was part of this validation pass.

## Retuning and model-upload memory, 11 October 2026

The subsequent [memory ownership audit](MEMORY_USAGE.md) extends this standalone
accounting to the direct game adapter and external transports. Serial refinement
passes share workspace and immutable model storage while retaining separate
histories, including one fence-protected upload on the conventional staged path.

The preserved historical gain survives the readable code: a fresh same-host
comparison measures 4.176840 ms for untuned kernels,
3.853808 ms for the saved optimized mix,
3.848888 ms for readable retuned kernels and
3.877672 ms for NVIDIA. This is 7.85% less GPU
time than untuned kernels on RTX 5080 / driver 616.64 at 1920x1080. Eight
isolated 128-frame processes per route exclude the first 32 frames; these are
complete GPU evaluation times, not game FPS. Ryzen 9 9950X3D, 16 cores / 32
threads, 96 GiB installed RAM.

324 readable settings and the retained 198-module historical pool informed
401 single-replacement screens. Six mixes, sixteen removals and two smaller
mixes were checked. The selected sixteen changes reduce GPU time by another
0.22% against the recovered readable baseline in independent counterbalanced
checks. Register limits and five selected assembler O1 settings reproduce the
confirmed whole-model mix; no kernel bodies, stages or precision were reverted.
A reduced-spill mix was slower. Static spill sizes are not total GPU-memory
allocations. All alternatives remain outside the production catalogue.

Immutable model weights now use GPU-upload heaps when the selected device
reports support. CPU writes complete before submission, avoiding a second
140.9375 MiB allocation. Unsupported devices use the staged path, whose upload
is retired only after a queue fence completes. Forced fallback and queued
retirement checks pass. Heap accounting separates default, GPU-upload, ordinary
upload and readback resources from DXGI residency.

At 1080p, owned resources are 203.125 MiB workspace, 140.9375 MiB weights,
33.75 MiB history/output and 31.875 MiB normalized inputs: 409.6875 MiB.
Mapped upload removes 140.9375 MiB from owned startup peak. Four counterbalanced
before/after processes show GPU local residency 715.63 to
574.70 MiB, private commit 1087.64
to 919.33 MiB, and resident working set
359.26 to 331.12 MiB. These process
figures include driver and offline harness storage; private commit is not
resident RAM. NVIDIA measures 686.52
MiB local residency and 351.03 MiB
working set in its standalone probe; its internal split is unknown.

All four saved extents and three additional dark/HDR/motion fixtures match
reset/final hashes, with zero debug errors and complete resource/module
retirement. Intermediate outputs and live game quality were not checked.
All 41 rebuilt production instruction/constant sections match tested modules.
Eleven C++ CPU suites, one WARP suite, 28 Python tests and all 118 embedded
payloads pass. Backup tag: `backup/native-before-retuning-20261011`.
No game deployment was performed.
