# Phoenix neural rendering implementation and acceptance

Updated 2026-10-18. Source implementations below are in the current experimental
candidate. The unchanged SDK remains available for before/after comparison.
The investigation baseline is preserved in
[implementation-audit.json](../../research/nvidia-dlss5/implementation-audit.json);
its source hashes describe the initial investigation, not today's dirty tree.

## Results

![Actual GPU output](../artifacts/current/neural-comparison/comparison.png)

![Portable temporal difference](../artifacts/current/neural-comparison/temporal-comparison.png)

Measured on RTX 5080 with a 384×256 SDR-proxy photo and a bounded three-frame
translation, full graph, all eight central blocks once, one network pass:

- Native V1 candidate versus unchanged native SDK: **exact on all three frames**.
- Explicit portable legacy profile versus unchanged SDK: **exact on all three frames**.
- New portable CUDA versus Vulkan: **exact on all three frames**.
- Corrected portable frame 3 versus old unit-gain output: mean absolute
  difference **0.00263437**, maximum **0.06574398** in 0..1 RGB. This measures a
  numerical migration, not a demonstrated quality improvement.
- Portable map-R-zero and protected regions: exact original RGB. Native versions
  pass through half textures, with maximum source difference **0.000243187**;
  resized 320×320 case maximum **0.000244141**.
- Identity field: exact versus each route's unmapped default. Opaque white HUD
  pixels: exact. Portable UI overlay versus alpha/backbuffer: exact; native
  maximum **0.000488281** after half backbuffer conversion/publication.
- Every optional plane traversed local x64, x64 broker and x86-to-x64 helper
  with identical output hash. Standalone file replay through all three routes also
  matched every optional-input hash and the protected output exactly. Rejected
  mapped frames preserved continuation.
- Invalid model switch preserves continuation; valid same-model replacement
  resets history/noise. Host/device map packing and bounded input/output alias
  checks passed with real full-model GPU execution.

These are Phoenix-to-Phoenix, algebra and contract checks. No NVIDIA-output
comparison, game rendering, arbitrary HDR UI, or cross-vendor acceptance is
implied. The native and portable implementations are separate numerical
pipelines; neither inherits the other's parity status.

## Development list

Checked source work is distinguished from remaining acceptance. All 16 areas
have been investigated; unavailable vendor data and untested integration are
kept explicit instead of inventing working provider/model options.

| Task | Source implementation completed | Evidence / remaining acceptance |
| --- | --- | --- |
| NR01 Baseline/capabilities | [x] Separate old SDK references, source/binary/model identities, capability JSON, stage/input/final hashes, actual GPU comparisons. | Full vendor-reference stage capture and each platform's color/graph/kernel identity remain open. |
| NR02 Temporal gain | [x] Read selected blend tensor before discard; shared gate logic/status; explicit legacy-unit mode. | Three-frame old-mode compatibility exact; corrected mode measured. NVIDIA nonzero-history reference still needed. |
| NR03 Sampling/precision | [x] Fixed/advancing seeds, uint32 wrap, per-pass counters and accepted-frame commit; half neural publication. | Rejected-frame continuation passed with advancing seeds. Vendor noise/rounding equivalence remains unverified. |
| NR04 Map contract | [x] Explicit experimental RGB/1 contract; retained preparation kernel G/B traced; A reserved1; range/stride/extent validation. | Independent NVIDIA per-channel/alpha/range/filter acceptance unavailable. Do not label Phoenix alpha semantics as NVIDIA's. |
| NR05 Versioned APIs | [x] Portable V2/V3, native D3D12/host V2, additive feeder V2, internal executor plane/evidence fields; old APIs retained. | C ABI x64/x86 and rejection/alias/continuation tests pass. Standalone native advanced path currently requires equal source/neural/output extents and refuses explicit depth/validity. |
| NR06 Portable maps | [x] Host/shared device/Vulkan feature packing, upload ownership and accounting, R display composition. | Vulkan host/device full-model exact; CUDA/Vulkan exact. HIP/SYCL/Metal/mobile must each run new mapped fixtures. |
| NR07 Native maps | [x] Retained descriptor byte32 and rectangle plumbing, stable GPU target, state/device/lifetime checks; standalone allocation lazy. | Native GPU mask/style/UI/field fixtures passed. Real moving engine objects and vendor map acceptance still needed. |
| NR08 Appearance | [x] Resource-guarded recovered style tables, neural conditioning, half publication and separate native/portable display stage. | Actual style outputs and intensity branches checked. NVIDIA grade/activation/transfer parity remains open. |
| NR09 Transport/capture | [x] Broker2 snapshots of all planes; V1 compatibility; worker owned vectors; asynchronous map/field capture; standalone V3 replay inputs and hashes; direct optional-plane fence capture with frame/generation/hash manifest. | Local/helper/WOW64-client outputs exact; WARP async display/capture and producer epoch checks passed; worker snapshots survive caller-plane mutation on both host bitnesses. Existing legacy binary replay probes deliberately refuse optional-plane captures; fixed SR-bias wire protocol remains separate. |
| NR10 Object producer | [x] Offline ID-to-map producer/preview/hash manifest; engine GPU ID-table shader; feeder/panel producer status. | Needs an actual engine's object/material IDs and moving/occluded object acceptance. Final color cannot identify arbitrary objects reliably by itself. |
| NR11 UI | [x] Premultiplied UI, alpha/backbuffer, original-domain display composition and UI-free recurrence; helper display uses current overlays. | Portable/native GPU HUD checks and software-D3D12 wrapper capture passed. HDR transfer and live game engine composition remain open. |
| NR12 Distortion | [x] Explicit two-direction normalized field, canonical scene/guides/map, previous-field motion conversion, inverse output sample. | Identity/translation full-model fixtures, synthetic dynamic field/edge checks. Curved time-varying vendor reference, scaled planes and game acceptance remain open. |
| NR13 Model lifecycle | [x] Resource identity, known-style guard, compatible portable model replacement and native resize/model reconfiguration; invalid replacements preserve session. | GPU invalid/valid switch and resize passed. Curated A/B/C tensors/metadata and arbitrary architecture importer are not available/implemented. |
| NR14 Streamline | [x] Feature1004 / exact frame-viewport tags50/69/70/71/72 harvesting, explicit-null retirement and bounded retention. | Public inspected headers declare tags; complete NR provider/header/plugin contract absent from inspected package. Actual Streamline NR evaluation/takeover cannot be certified from tag observation. |
| NR15 Capability/UI | [x] Model-free discovery, route profiles, effective gain/packing/status, producer/map/UI/field/revision and separate persisted Phoenix intensity/style controls in shared panel; unsupported requests reject. | Addon compiles; live panel/engine resource lifecycle needs a connected producer. Existing NVIDIA-only labels still describe their original controls. |
| NR16 Acceptance | [x] Full-model GPU before/after, portable/native/CUDA/Vulkan outputs, x86 helper, bounded transaction/resize tests, C/CPU/operator contracts, software-D3D12 display/capture. | Vendor image parity, long moving-object/curved-field sequences, GPU peaks and game performance matrix remain acceptance work. Tiny fixture timings are not game FPS. |

## Timing and validation scope

The final native V1 compatibility run took 15.93 ms cold and 4.11/3.90 ms warm
including host transfers at 384x256. The final portable Vulkan run took
356.77 ms cold and 16.50/17.46 ms warm. New native feature calls in the separate
reset-case run took 4.52-7.90 ms after the first 44.34 ms call; portable ones
took 18.90-22.46 ms after the first 361.22 ms call. These are individual measured
calls on a small photo fixture, not statistically robust benchmarks or game FPS.
Cold setup, host transfer and shader compilation are included in call latency;
internal GPU timing has its separately reported scope. Allocation ledgers are
not whole-process VRAM or temporary-peak measurements.

The model DLL was read as data. No game was launched or changed, no NVIDIA DLL
code executed, and the frozen reference was untouched. The old SDK/baseline
outputs were not regenerated from candidate expectations. Win32 GPU execution
was verified through the x64 helper; a direct Win32 Vulkan build encountered
stdcall import-library linkage errors and is not included as a validated route.

The candidate is in `artifacts/current/neural-candidate`; it is model-free and
points to the existing local model/kernel bundle. Receipts and runnable fixture
jobs are in `artifacts/current/neural-comparison`. Recursive temporary-build
cleanup was rejected by automatic approval review; the two verified project
build folders remain. The tool gave only "blocked by policy" as its reason.

## Remaining development after this candidate

1. Connect a real engine ID/material producer to feeder V2, including moving
   edges, occlusions, resize/eye/settings generations and frame freshness.
2. Extend advanced native scaled-plane/HDR/depth-validity contracts with actual
   producer transfer/coordinate evidence; current requests reject those cases.
3. Add optional-plane consumption to the older direct binary replay probes;
   they now reject expanded captures safely. Standalone V3 plane-file replay is
   already available.
4. Run the mapped/style/UI/field acceptance matrix on HIP, SYCL, Metal and mobile
   hardware; shared source is not execution evidence. Repair/test direct Win32
   Vulkan imports if that route is needed beyond the working x64 helper.
5. Obtain NVIDIA reference outputs and the complete public/authorized NR provider
   API, plus any curated appearance-model resources. Compare stage boundaries,
   unknown mask alpha, grade activation, filtered motion and noise sequences.
6. Measure longer motion/curved-field clips, whole-call latency, allocation peaks
   and actual games only against declared references and acceptance bounds.

## Computation and iteration

![Pipeline](../artifacts/current/neural-comparison/pipeline.png)

Phoenix already executes the multiscale encoder, central attention blocks,
decoder skips, RGB/gate head and temporal history. Repeating a central block,
executing another whole-network pass, and carrying history to another video
frame are separate mechanisms. There is no missing ComfyUI-style diffusion
sampler/VAE loop to add. Normals/albedo/lighting described as training supervision
are not automatically missing runtime planes. Individual blocks' learned
semantic roles cannot be established without ablations.

Exact formats, lifetime rules, output arithmetic and library calls are in
[LIBRARY.md](LIBRARY.md) and [NATIVE_LIBRARY.md](NATIVE_LIBRARY.md). Compact
machine-readable evidence is in [summary.json](../artifacts/current/neural-comparison/summary.json)
and [neural-rendering-development.json](../artifacts/current/neural-rendering-development.json).

## Performance evaluation follow-up

The full [25-resolution performance evaluation](NEURAL_RENDERING_PERFORMANCE.md) is complete. Native timings completed at 24 sizes, with 122/125 portable cases completing; failures are retained rather than extrapolated. These findings add concrete acceptance work:

- [x] P0: diagnose and fix native 1024x768 first-frame failure, present in both baseline and candidate. Preserve the failing fixture and report the actual D3D12/device error.
- [x] P0: fix native reset repeatability at 3840x1600 (large error) and 3840x2160 (smaller numerical variation), present in both versions. Require reset/continuation numerical acceptance before calling these sizes validated.
- [ ] P1: profile the 5120-wide timing cliffs/variation using live per-process GPU activity and longer steady-state runs; do not infer causation from stale NVML process entries or attribute baseline variation to the new controls.
- [ ] P1: move portable field/grade/UI composition to GPU operations; investigate the advanced API's empty-plane serial work and bounded snapshot/preflight overhead. Keep numerical reference fixtures.
- [ ] P1: add graph/device-budget preflight and lower portable transient GPU memory. 8K Vulkan cannot allocate; CUDA requests about 16 GiB for backend buffers alone and runs under memory pressure.
- [ ] P2: make advanced native buffer allocation more selective and document/review retained caches. Full suite tracked memory rises from 403 to 563 MiB at 1080p and from 3.82 to 6.18 GiB at 8K, excluding caller textures and driver caches.


## Add-on feature regression check - 2026-10-18

[Current test report](../artifacts/current/neural-tests/REPORT.md): x64/x86
runtime contracts, 20 add-on CPU suites, two software D3D12 suites, 105 Python
tests, 2,774 real panel widget checks and ten full-model GPU feature/transport
groups passed. New appearance click tests cover intensity, Natural/Cinematic,
separate NVIDIA values, persisted tuning and enable/disable restoration.

Targeted native correctness checks still fail at 1024x768 on the first frame
and at 3840x2160 on repeated explicit resets, for both the packaged candidate
and fresh native build. 3840x1600 passed this run; its earlier intermittent
failure remains open. No game or live engine producer was exercised. The
packaged candidate is unchanged; tests used the hash-pinned packaged helper.
An unrelated host executable source changed outside this task after the add-on
build and is outside its compiled dependency closure; this run does not validate
that changed host source.


## Native correctness repair - 2026-10-18

The two native P0 items above are fixed in the current experimental candidate.
[Diagnosis and evidence](../artifacts/current/native-correctness/REPORT.md): the
model-free kernel bundle retained an obsolete graph, despite the corrected
source graph already existing. Its 32-token tail storage was undersized; at
1024x768 the last projection scratch extended beyond the workspace allocation.
The versioned, hash-pinned graph now reaches the current runtime and packaging
rejects stale graph data. Obsolete layouts fail before native graph execution.

Eleven resolutions including immediate 1024x768/4K neighbors and 8K passed
209 full-model frames with exact resets; the packaged candidate passed another
76 frames in two fresh sessions per failing size. Nine native advanced-feature
fixtures, 15 runtime contract tests and 31 Python tests passed. All 41 kernel
binaries and model weights are unchanged. Earlier performance and feature-test
reports describe their recorded pre-fix hashes; no timing sweep was repeated.
No game was launched or installed. Broader engine/HDR/temporal acceptance remains
separate from these synthetic correctness fixtures.
