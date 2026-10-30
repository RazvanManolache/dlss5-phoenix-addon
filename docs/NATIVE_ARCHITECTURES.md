# Native RTX architecture variants

The current source builds architecture-specific Phoenix kernels for both
**Original** and **Lightweight FP8**. This is separate from choosing a community
NVIDIA runtime: these kernels execute through Phoenix's own Native RTX backend.
Model resources, packed storage, graph topology and the 41-module launch ABI stay
the same. Native model folders and supported NVIDIA DLL weight sources remain
available.

The latest [native Phoenix DLSS5 / NVIDIA DLSS5 endpoint comparison](benchmarks/current/README.md)
uses the tuned bundle and a matched structured fixture. The before/after tuning
studies below retain their gradient/detail fixture and separate measurement
identities; their timings should not be spliced into the endpoint comparison.

| Target | Consumer RTX generation | Arithmetic / memory path | Validation boundary |
| --- | --- | --- | --- |
| `sm_75` | RTX 20 | Packed E4M3 decoded to FP16 Tensor Core operations, split into k8 steps; aligned 16-byte synchronous tile copies | Experimental; compile/offline checks, no older-device inference |
| `sm_86` | RTX 30 | Packed E4M3 decoded to paired k16 FP16 operations; aligned 16-byte ordinary `cp.async` tile copies | Experimental; compile/offline checks, no older-device inference |
| `sm_89` | RTX 40 | Native FP8 matrix arithmetic; aligned 16-byte ordinary `cp.async` tile copies | Experimental; compile/offline checks, no older-device inference |
| `sm_120` | RTX 50 | Native FP8 Tensor Core, bulk-copy/barrier and vector-atomic kernels, with wider cached reads | Both models pass exact before/after reset/history checks on RTX 5080 at 320×180, 801×603 and 1080p; the current tuned images are rebuilt, not claimed byte-identical to the earlier SDK |

The driver selects an architecture image when creating each NVAPI module; no
per-frame CPU/GPU architecture branch is added. Only the chosen code executes.
The multi-architecture binaries contain cubins, not fallback PTX that could be
silently JIT-compiled for an unsupported GPU.

## Automatic selection in the experimental addon

### Selectable older instruction styles

**Setup > Device & compatibility > Processing device > Native RTX kernel style**
defaults to **Automatic (native, recommended)**. A style-enabled experimental
bundle additionally offers the following lower-capability strategies:

| Physical card | Default | Optional older styles |
| --- | --- | --- |
| RTX 20 | Native RTX 20 | None |
| RTX 30 | Native RTX 30 | RTX 20-style |
| RTX 40 | Native RTX 40 | RTX 30-style, RTX 20-style |
| RTX 50 | Native RTX 50 | RTX 40-style, RTX 30-style, RTX 20-style |

Every image is compiled for the **actual** GPU target. The style controls the
FP8/FP16 matrix decomposition, conversion, copies, barriers and atomics selected
at compile time. This does not emulate an older GPU's memory system, scheduling,
cache or throughput. Benchmark results measure the strategy on the physical
card, not predicted performance of an RTX 20/30/40. Generic compiler lowering
still targets the physical GPU.

Styles share Original/Lightweight weights and the native launch ABI. Only the
selected kernel catalogue is loaded. There is no per-frame style dispatch and
no second weight allocation just to offer the menu. Changing a style creates a
new executor after current work retires; unavailable styles are disabled. A
refused same-process live change retains the previous executor; helper routes
use their existing restart/recovery handling. Settings, presets, helper
requests and timing records retain `NativeKernelStyle` (`auto`, `rtx20`, `rtx30`,
`rtx40`). NVIDIA DLSS5 DLL and portable executors do not use this setting.

The native addon and its x64 graphics helper support selection. The experimental
WOW64 library/driver bridges require Automatic and report that restriction,
allowing the existing helper fallback rather than silently ignoring a style.

Build a model-free style bundle without replacing the existing native images:

```powershell
python tools/build_kernel_styles.py --base-native <existing-native-bundle> `
  --compiler <CUDA-bin/nvcc.exe> --jobs 4
```

The current result goes to `artifacts/current/native-kernel-styles`; compiler
staging is removed on success or failure. Build the addon with this native
bundle and `--experimental-native`. New styles retain pending device-inference
validation until explicitly tested; successful compilation is not runtime proof.

Phoenix 0.12 publishes this path as the separate **RTX Compatibility Windows**
download, for both x86 and x64. The standard Windows download keeps the validated
single-target RTX 50 payload. See [package selection](RELEASE_0.12.md); install
only one variant. The stable embedding gate is not relaxed by the public experiment.

Select Phoenix with the Native RTX executor as usual; there is no separate
older-card switch. Each packaged module contains all four targets. Phoenix
passes the module and the active D3D12 device to NVAPI, and the driver selects
the compatible image for that device when creating the module:

- RTX 20 uses `sm_75` compatibility kernels.
- RTX 30 uses `sm_86` compatibility kernels.
- RTX 40 uses `sm_89` kernels.
- RTX 50 uses the native `sm_120` kernels.

Both Original and Lightweight use this same automatic loading path. Selection
uses the processing device rather than the first GPU name found on the machine,
so a helper processing on another adapter uses that adapter's compatible image.
Phoenix does not redirect the NVIDIA DLSS5 backend to its own kernels; that
backend continues to execute the selected NVIDIA runtime.

Pending older-card validation does not disable these paths in the explicit
experimental package. Unsupported hardware or a driver that rejects module
loading still reports a startup failure through the existing route handling;
Phoenix does not force a mismatched architecture image onto the GPU.

## Arithmetic and synchronization

`common/architecture_math.cuh` supplies the pre-Ada conversion and matrix helpers.
Small integer conversions implement saturating E4M3 with nearest-even rounding,
signed zero and NaN handling, preserving the packed two-byte interface. An FP8
k32 operation is decomposed into two FP16 k16 operations; Turing further splits
each into two k8 operations. This follows OpenNR's compatibility approach, with
attribution retained in [THIRD_PARTY.md](../THIRD_PARTY.md). It does not retrain or
expand the installed model weights.

Compatibility tiles previously used dependent four-byte loads/stores in one
elected lane. `common/compatibility_copy.cuh` handles aligned requests in 16-byte
units: synchronous vector loads/stores for Turing, ordinary `cp.async.cg` for
Ampere/Ada. Ampere waits in the issuing lane before returning. Ada waits at the
existing consumer barrier, allowing independent setup to overlap the transfer;
every thread then reaches the same block barrier. This is not Hopper's bulk-copy/barrier
instruction. No warp collective is introduced inside the divergent leader branch.
Unaligned/tail requests and bounds-checking launches retain the scalar reference;
requests are never rounded up. The same helpers serve older native images and
older styles compiled for newer cards.

The pre-Ada GPU decoder uses an exponent-bias adjustment and a register permutation
for the eight zero/subnormal encodings, avoiding divergent scalar decoding.
Signed zeros and canonical signed NaNs match the scalar reference for all 256
codes. The encoder, packed storage and matrix accumulation order stay unchanged.
Four packed-half atomic updates replace unsupported vector atomics; they remain
atomic and retain their address/update order. RTX 50 keeps native arithmetic,
bulk-copy synchronization and atomics; its read optimization is described below.
Different matrix shapes can still change rounding relative to native FP8; actual
older-card inference, image quality and performance require device testing.

### Compatibility optimization check at 1080p

This first optimization comparison ran both full models on one RTX 5080.
Values are GPU milliseconds per processed frame, taking the median of three
fresh-process medians per model/style. Each process warms for one second and
measures for at least three seconds, with alternating style order. The older,
very slow Original paths reached the existing 15-second measurement cap after
27 samples; optimized runs meet the 30-sample goal.

| Style on RTX 5080 | Original before → after | Speedup | Lightweight before → after | Speedup |
| --- | ---: | ---: | ---: | ---: |
| RTX 20-style | 554.929 → 41.366 ms | 13.4× | 328.942 → 23.803 ms | 13.8× |
| RTX 30-style | 554.152 → 39.528 ms | 14.0× | 336.233 → 22.898 ms | 14.7× |
| RTX 40-style | 40.564 → 4.692 ms | 8.6× | 31.525 → 2.907 ms | 10.8× |

Automatic/native remains preferred: its unchanged RTX 50 images measured about
3.86 ms for Original and 2.29 ms for Lightweight in the new run. All 24 optimized
processes produced finite, reset-repeatable outputs matching the corresponding
pre-change style's reset/history captures exactly. This preserves each fallback's
existing numerics; it does not make 20/30-style accumulation identical to native
FP8. The deterministic gradient/detail fixture uses zero motion, default controls,
one full pass and all eight transformer blocks. GPU timestamps include matching
output copy/barriers, excluding cold loading, producer upload, readback and CPU
gaps. These are not gameplay FPS or predicted results for older physical cards.

An alternate branchless encoder won an isolated microbenchmark but showed no
full-model gain in a boundary-kernel spot check, so it was not adopted. Single-
launch diagnostic timestamps locate expensive stages but perturb execution;
only the uninstrumented runs above support the end-to-end timing comparison.

Reproduce the focused GPU checks explicitly:

```powershell
python tools/test_compatibility_copy.py --gpu --output <copy-receipt.json>
python tools/test_native_architecture_arithmetic.py --gpu --output <arithmetic-receipt.json>
python tools/benchmark_kernel_styles.py --bundle <style-bundle> --model-dll <model-dll>
```

Copy checks cover 288 cases per style, including elected-lane divergence,
unaligned requests, zero lengths, tails and untouched sentinels. The decoder is
checked against the host reference for every FP8 code; the arithmetic checks also
exercise all 65,536 half patterns and both compatibility MMA shapes. GPU checks
here use RTX 5080. Offline inspection verifies all 164 native and 246 style images,
including byte-identical retention of all 41 RTX 50 native images. Older devices
retain their experimental/pending validation status.

### Further tuning: full-model confirmation

The next comparison uses the already optimized bundle above as its control.
It tests **all five proposed directions**, then rebuilds every physical target
and packages only the useful changes. Automatic/native remains preferred.

| Direction | Outcome |
| --- | --- |
| Predecode/reuse FP8 weights | Keep a **512-byte read-only global vocabulary per arithmetic module** for B/weight-side fragments on 20/30 paths. This caches the floating-point format, not expanded trained weights. A genuine FP16-expanded weight operator was faster in isolation but doubled that weight buffer and needed a separate prefill; it was not integrated. |
| Packed-pair decoding | Reject: arithmetic pair decoding was about 1–2% slower. A 256 KiB global pair vocabulary gave gains but cost more memory than the chosen byte vocabulary. Divergent `__constant__` lookup was much slower. |
| Copy/compute overlap | Keep deferred ordinary-copy waits for 40-style only. The 30-style experiment was flat and retains immediate completion. Native 50 bulk-copy barriers are unchanged. |
| Shorter fragment lifetimes / register pressure | Reject the general lifetime rewrite after full-model confirmation. Keep a targeted **160-register ceiling** in the pre-Ada 64-channel attention stage to avoid a measured cache-induced spill; 40/50 keep 128. Budgets 160/192/224/255 were tested for both compatibility styles. Larger budgets offered no dependable benefit. |
| Selective fusion / layout / native improvements | Keep aligned four-word reads as one 16-byte cached read on 40/50 paths. Reject a working producer/repack fusion prototype: it really removed one dispatch and matched outputs, but did not improve full-model time. Extra MMA ordering also gave no useful gain. No production graph or kernel ABI changes were adopted. |

Final **1920×1080**, full-graph, one-pass results on RTX 5080:

| Style | Original control → tuned GPU ms | Paired time reduction | Lightweight control → tuned GPU ms | Paired time reduction |
| --- | ---: | ---: | ---: | ---: |
| Automatic / native 50 | 3.857 → 3.799 | 1.35% | 2.282 → 2.232 | 2.13% |
| RTX 40-style | 4.700 → 4.491 | 4.45% | 2.907 → 2.766 | 4.84% |
| RTX 30-style | 40.449 → 34.613 | 12.61% | 22.992 → 20.747 | 9.82% |
| RTX 20-style | 42.207 → 37.770 | 10.55% | 23.917 → 21.890 | 8.43% |

Each 1080p condition uses three fresh processes, one second of warmup and two
seconds of measurement, with at least 30 frames. Before/after order alternates.
Times are separate medians of process medians; the percentage is the **median
of matched-round changes**, so it need not equal the ratio of those separate
medians. Earlier screens showed drift over minutes; paired comparisons reduce
that confound. All three final 1080p pairs improved for every row. These are
modest fixture gains for native 50, not an overall game-FPS promise.

Both models preserve the previous corresponding style's reset/history hashes
exactly, with finite, repeatable reset outputs at **1920×1080, 320×180 and
801×603**: 80 successful process cases. The same deterministic FP16 gradient/detail
fixture, zero motion, default controls and eight-block schedule are used. As
above, GPU timing includes the model and matching output copy, excluding model
loading, producer upload, readback and CPU gaps. That confirmation focused on
1080p; the subsequent [720p–8K sweep](NATIVE_TUNING_RESOLUTIONS.md) covers larger frames.

**Memory matters too.** The first lookup build increased Original-model scratch
memory by roughly 70–78 MiB despite the tiny table. Driver function attributes
identified a local-memory spill: the 64-channel stage grew from roughly 0.65 KiB
to 2 KiB per thread. The targeted register adjustment removed the process-level
increase. Against the control, the final 20-style runs use about **16.8 MiB less
device-local memory and 29–30 MiB less private RAM**; 30-style uses about
**12.6 MiB less device-local memory and 22–23 MiB less private RAM**, for both
models. Device-local usage is unchanged for 40/50; their small private-RAM
differences vary with harness/driver allocations. These are current-process DXGI
residency and RAM observations, not vendor-internal buffer accounting or cold
startup/peak-memory benchmarks. Model files and packed weight storage are unchanged.

All 164 native and 246 selectable-style device images pass offline checks.
The current rebuild intentionally changes 30 native RTX 50 image hashes; use
`test_native_architectures.py --allow-sm120-changes` to record those differences
while still verifying every image, entry and manifest pin. Default validation
continues to reject unexpected RTX 50 changes. Copy/decoder GPU checks cover
288 transfer cases, all 256 codes, all 65,536 code pairs and 131,072 weight-fragment
cases per tested style. Both compatibility MMA decompositions pass the independent
49-warp arithmetic check, including its deliberately corrupted negative control.

The maintained experiments are `tools/experiment_native_tuning.py`,
`tools/experiment_layout_fusion.py`, `tools/experiment_cache_registers.py` and
`tools/validate_tuning_primitives.py`. Historical source-transform recipes require
their earlier snapshot when an optimization is already present; they reject
changed anchors instead of silently timing the wrong experiment. Current compact
local receipts are under `artifacts/current/kernel-tuning/` (ignored by Git).

To compare any final candidate with a retained control explicitly:

```powershell
python tools/verify_native_tuning.py --baseline <control-bundle> `
  --candidate <tuned-bundle> --output <paired-receipt.json>
```

Only RTX 5080 was physically tested. The 20/30/40 results measure those
instruction strategies **on that card**, not predicted performance on older
cards. Older devices retain pending experimental status. These experiments
launch no game, change no installed game files and publish no release.

### Higher-resolution follow-up

The [full resolution report](NATIVE_TUNING_RESOLUTIONS.md) adds **1280×720,
2560×1080, 2560×1440, 3840×2160, 6144×3456 and 7680×4320** for both models
and all four styles. All 288 process cases preserve exact reset/history outputs.
The 20/30-style reductions remain about 8–13%; 40-style medians improve at every
size, ranging about 1–6%. These remain instruction-style results on RTX 5080.

Native 50 improvements are smaller and size-dependent: 2560×1080 is essentially
unchanged in this fixture. The first 8K three-pair comparison straddles zero;
a separate five-pair check with longer warmup measures about **1.1% for Original
and 1.4% for Lightweight**, with all ten pairs improving. Both sets of results
are retained rather than replacing the initial noisy values. This is useful
evidence for a modest change, not a universal game-FPS claim.

The candidate native-50 process uses about **4.8 GiB of device-local memory at
8K** for either model, including the harness and driver allocations. The report
also records memory readings at every size. The old control is reconstructed
in temporary storage and checked against saved 1080p outputs before the sweep;
native-50 code images retain their exact saved bytes. Recompiled compatibility
image hashes and both benchmark-probe revisions are recorded. All temporary
build/input folders are removed, and the installed/published addons are untouched.

## Building

The production builder defaults to all four architectures. CUDA 13.1 and the
existing MSVC setup were used for the offline checks. Build scripts use Python;
the deployed addon, kernels and model processing do not.

```powershell
python tools/production_build_native_kernels.py native_arch_build `
  --compiler 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe' `
  --jobs 4 --preserve-sm120 artifacts/current/sdk/windows-x64/native
```

`--preserve-sm120` verifies each baseline graph/launch contract and all 41 cubin
hashes before copying those RTX 50 binaries unchanged. Missing, altered or
incompatible baselines are rejected. Omit it to compile all targets from source;
use `--architectures sm_120` for the previous single-architecture output format.
Additive model-identity metadata is accepted during graph migration; execution
plans and existing identity fields must still agree.

Large compatibility kernels can take more than ten minutes to compile. The
per-command limit defaults to one hour and can be set with `--compile-timeout`
(seconds). After an interrupted build, repeat the command with `--resume`.
Resuming checks the recorded source/graph snapshot, build recipe and CUDA image
before reusing a completed kernel. Source or graph changes require a fresh build.

Each combined module has all four architecture images and a SHA-256 pin.
`architecture_images` records each image's target, size and hash. The manifest
records `selection_status=compiled`, `validation_pending=true` and per-architecture
pending inference status. Compilation never grants inference validation or bypasses
the stable addon embedding gate. The saved-hash validator handles single-architecture
bundles; one GPU cannot certify the other images. Stable packaging requires every
declared target's device validation state to be `validated`.

An explicit `tools/build_addon.py --experimental-native` build can embed a
successfully compiled, model-free bundle while retaining its pending architecture
validation states. The addon identifies itself as experimental in its exported
name and overlay. The stable default remains strict.

To build and test the experimental Windows x64 package from an existing validated
addon runtime and the new kernels:

```powershell
python tools/build_experimental_rtx_addon.py `
  --source-addon artifacts/current/addon/Phoenix.addon64 `
  --native build/native_arch_build
```

This command performs GPU tests. It writes the current ZIP, standalone addon,
validation receipt and checksums under `artifacts/current/addon-experimental-rtx`.
The installed addon needs no Python, compiler or CUDA SDK. Model files remain
external. Published addon/SDK files are not replaced by these local builds.

## Verification

```powershell
python tools/test_native_architectures.py `
  --bundle build/native_arch_build `
  --baseline artifacts/current/sdk/windows-x64/native `
  --model '../research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll' `
  --output artifacts/current/native-architectures/validation.json
```

This runs CPU/offline checks only: CUDA's binary extractor independently unpacks
all 164 images, verifies their architecture/entry/hash and checks all 41 RTX 50
images against the baseline. An independent grid oracle exercises all 65,536 half
bit patterns and all 256 E4M3 codes through the compatibility converters. Real
model graph/launch tests use injected recording functions instead of NVAPI/GPU
execution. No game, installed configuration or frozen model reference is changed.

An explicitly requested arithmetic device check can compare both compatibility
MMA decompositions with native FP8 on an Ada or newer GPU:

```powershell
python tools/test_native_architecture_arithmetic.py --gpu `
  --output artifacts/current/native-architectures/arithmetic.json
```

The fixed hashed operands use exactly representable products/sums to distinguish
fragment mapping from expected reduction-rounding differences. Both decompositions
match native FP8 in 49 warp cases (1,568 lane results each) on RTX 5080. A deliberately
corrupted operand must fail the same test. This exercises compatibility arithmetic
on the current GPU; it does not validate an older card, its driver or full-model
performance. Omit `--gpu` to run CPU conversion checks only;
`--gpu-architecture sm_89` builds the arithmetic probe for Ada.

## Initial multi-architecture validation

The complete four-target build passed independent extraction checks for all 164
device images. CPU model/plan checks passed 104,448 assertions, including 16
Lightweight dispatch-depth cases, 15 block-skip cases and 12 frame extents.
All 27 focused Python tests passed. Receipts:
[images and plans](current-native-architectures/validation.json),
[arithmetic](current-native-architectures/arithmetic.json),
[paired RTX 5080 comparison](current-native-architectures/paired.json).

| Phoenix model | Extent | Previous GPU ms | Combined-module GPU ms | Output |
| --- | --- | ---: | ---: | --- |
| Original | 320×180 | 1.852 | 1.850 | Exact reset/history match |
| Lightweight FP8 | 320×180 | 1.016 | 1.017 | Exact reset/history match |
| Original | 1920×1080 | 3.843 | 3.837 | Exact reset/history match |
| Lightweight FP8 | 1920×1080 | 2.271 | 2.270 | Exact reset/history match |

These are deterministic synthetic-input fixtures on RTX 5080, not gameplay FPS.
1080p values are medians of three process medians with alternating before/after
order; 320×180 uses one pair. Each process warms up for 0.5 seconds and measures
for one second. Captures also match across repeated processes. Changes below
0.2% are within the observed timing variation, not a speedup claim.

Combined modules occupy 22.67 MiB versus 3.40 MiB for the single-target baseline.
Startup latency and peak host memory were not benchmarked. Older-card full-model
inference, quality and timings remain unvalidated, and the generated bundle
retains its compiled/pending-release-validation status. Published packages and
the validated SDK baseline remain intact.

Before promoting the experimental addon to stable, complete release validation
and test reset/history behavior, numerical/visual results and timings on each older
generation. Driver support for the NVAPI module-loading interface is
also required. These source targets do not certify every GPU sharing a series
name, non-RTX cards, or a particular older driver.

## Initial packaged addon verification

The local experimental Windows x64 addon passed all 22 CPU suites and both
software D3D12 suites. A minimal ReShade host loaded the actual DLL, checked its
experimental identity and feeder ABI exports, and verified balanced registration
and unloading. The production extraction code verified all 109 embedded files,
including decompression and hashes, in a temporary cache. No NVIDIA model DLL is
included. These checks do not validate retail-game presentation.

Both models then ran on the RTX 5080 using the kernels and forwarding shim
extracted from that actual addon. All 41 embedded `sm_120` images match the SDK
baseline byte for byte. Reset/history captures match exactly at both extents.

| Phoenix model | Extent | Baseline GPU ms | Packaged GPU ms |
| --- | --- | ---: | ---: |
| Original | 320×180 | 1.848800 | 1.847200 |
| Lightweight FP8 | 320×180 | 1.013536 | 1.015808 |
| Original | 1920×1080 | 3.828864 | 3.829632 |
| Lightweight FP8 | 1920×1080 | 2.261312 | 2.262016 |

Each case uses one paired process comparison with 0.5 seconds of warmup and one
second of measurement. These synthetic fixtures show essentially unchanged
timings; they are not gameplay FPS or a statistical speedup claim. Older RTX
20/30/40 device inference remains untested. That development run changed no game
files and uploaded no release. The [packaged-addon receipt](current-native-architectures/addon-validation.json)
records the exact addon hash, suite results, captures and embedded-resource audit.
