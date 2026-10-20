# Phoenix DLSS5 Multi Model Agent Handoff

For the optional native exports, central model locations and DLSS Everything
integration, read [Shared model library handoff](HANDOFF_SHARED_MODEL_LIBRARY.md)
(verified 20 October 2026).

<!-- current-two-model-evidence -->
Current follow-up: [support matrix](SUPPORT.md) and [two-model benchmarks](benchmarks/current/README.md). Fresh isolated measurements cover native Phoenix/NVIDIA Original and Lightweight, CUDA, Vulkan NVIDIA/AMD and CPU. Metal/HIP/SYCL are implemented but unavailable for fresh timing on this host. Raw receipts are under `artifacts/current/readme-benchmarks`; existing extraction and portrait results are preserved.

RTX Vulkan measurements use `vulkan_fused_feed_forward: false`: the automatic fused FFN failed 4K reset repeatability in both models. The default optimized-path issue remains unresolved; full trained execution through the ordered pair is the checked workaround.
<!-- /current-two-model-evidence -->

Status: 19 October 2026. Use this before extending the new model support.
All paths below are relative to this repository root unless stated otherwise.

## Current Result

NVIDIA DLSS5 Neural Rendering DLL **310.9.16.0 contains two models**, not just a
replacement for the old one. Phoenix DLSS5 can now read the unchanged original
model from either DLL, including through the native kernel-bundle loader.
The smaller model now has a separate **portable decoder and graph**, plus a
**78-launch native RTX schedule**. CPU inference and native plan/dispatch checks pass;
native GPU timings/reset/output checks now pass at 720p, 1080p and 4K, with
[matched NVIDIA comparisons](../artifacts/current/four-model-comparison/README.md).
Both models match corresponding NVIDIA reset/history captures byte-for-byte on
these fixtures, with controls explicitly matched. General parity is not established. See [SMALL_FP8.md](SMALL_FP8.md)
and `artifacts/current/small-model-support/` for the current implementation and receipts.
The earlier validation section below records the initial registry refactor.

The multi-model registry and loading refactors are source changes, not a released
or deployed update. The checkout contains substantial unrelated uncommitted work.
Inspect the current tree; do not reset it or assume HEAD represents these changes.

The FP8 runtime has subsequently been refactored for readability. The portable
recorder has named stage helpers and explicit stage/skip tables; the native
adapter separates selection, remapping, shift patches and lifetimes. Its
pre-refactor graph/weight fingerprints are retained as a regression fixture in
`tests/fixtures/small_model_graph_fingerprints.json`. Before/after and compiled
CPU validation receipts are in `artifacts/current/small-model-support/refactor/`.
This cleanup does not alter the recovered research files, original-model source,
kernel arithmetic or weights, and it has not been deployed or GPU-benchmarked again.

## DLLs and Evidence

The new DLL was obtained from the user's downloaded Immersive Visuals 1.0.7
package. Only the NR DLL was extracted, into a separate research location.
Its NVIDIA Authenticode signature was valid at inspection. No downloaded code
was executed to inspect or extract its resources.

| Container | Local file relative to repository root | Bytes |
| --- | --- | ---: |
| Original 310.8.0.0 | `../_dlls/nvngx_dlssnr.dll` | 165,840,496 |
| New 310.9.16.0 | `../research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll` | 266,424,944 |

Whole-file SHA-256 values:

```text
310.8.0.0   e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e
310.9.16.0  2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f
```

| Resource | Model ID | Serialized records | Resource bytes | Implementation |
| --- | --- | ---: | ---: | --- |
| `WEIGHTS_HT` | `original` | 153 | 147,695,410 | Original decoder and graph |
| `WEIGHTS_CC_SMALL_FP8` | `small-fp8` | 77 | 113,893,343 | Portable decoder and graph |

`WEIGHTS_HT` is byte-identical in both DLLs. All 153 payload hashes and all
658 decoded tensor hashes matched. Do not confuse serialized records, decoded
tensors and network layers; they are different quantities.

Useful fingerprints:

```text
Original raw resource     836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4
Smaller raw resource      f743e8eb320f42bc9bdc992ee2bf51f7bb7c8688ced4ecf5ce6807ec4eef7962
Original layout           531a2a237808c9a5a1b040b506a5cd480caa497f6725ec595ae63f5891771680
Smaller layout            f16f2ef91fc207eb384f0db91a627766bd853fd8013b1f532d8ea6bd64c150fd
Original tensor catalogue db9949368cd15e17886ec5f134e117c1de52889db6db6e4dd59f456bd702743e
Original native packing   a5513b1845c98a486985ed04f38e66a1854cce33c2aba3a505866028bd4ee3e5
```

The layout digest covers sorted record names and payload sizes, not learned values.
The tensor catalogue digest covers sorted decoded tensor names, sizes and hashes.
The native packing digest covers the complete lexically ordered, 512-byte-aligned
allocation, including padding. These hashes are not interchangeable.

Research evidence is in the sibling directory `../research/model-compatibility-310.9.16/`:
`resource-comparison.json`, `loader-probe.json`, and inspection/probe sources.
**Its README and loader probe describe the initial pre-refactor state.** Their
statements about the old hardcoded loader and native DLL rejection are historical.
Use current source and [MULTI_MODEL.md](MULTI_MODEL.md) for current behavior.

## Model Registry and Dispatch

The registry has no two-model limit. It is a compiled registry, not an arbitrary
JSON/plugin loader. Future models require explicit registration and validation.

| Source | Responsibility |
| --- | --- |
| `src/native/model/definition.h` | Stable ID, architecture/resource/layout, counts, history-gain block, availability and reference hashes |
| `src/native/model/registry.h` | Arbitrary-length definition table; lookup and duplicate/incomplete-definition rejection |
| `src/native/model/models/original/definition.h` | Original model metadata |
| `src/native/model/models/small_fp8/definition.h` | Smaller-model metadata, 302 tensors, portable implementation available |
| `src/native/model/weight_source.h` | Configuration, authenticated reads, selected-layout validation, packing and resource identity |
| `src/native/model/portable_model_io.h` | Bounds-checked named PE-resource lookup without executing DLL code |
| `src/native/model/resource_identity.h` | Native resource compatibility and separate picker container policy |
| `native_runtime/model.h` | Common tensor access and architecture-keyed decoder table |
| `native_runtime/graph.h` | Shared operations and architecture-keyed graph-builder table |

Model and architecture IDs are strings; there is no two-value architecture enum.
Multiple compatible weight editions can share one architecture implementation.
Add a new architecture's handlers to `Model::decoder_for` and `Graph::builder_for`.
Missing handlers fail explicitly, even if a definition is mistakenly marked implemented.
Registry string views must refer to sufficiently long-lived storage; built-ins use
static definitions. Do not register dangling views into temporary parsed JSON strings.

Original-specific implementation is isolated under:

```text
native_runtime/models/original/decoder.inc
native_runtime/models/original/weight_layouts.h
native_runtime/models/original/graph.inc
```

The `.inc` files are included inside their owning classes. The original graph-recording
body was moved without changing its operations. `native_runtime/weight_layouts.h`
remains a compatibility include. Do not replace its original table with a new model's table.
The smaller runtime folder contains its own decoder entry, layout table and graph.
Shared packed decoding operations are under `native_runtime/models/common/decoder.inc`.

## Configuration and Loading

```json
{
  "weight_source": {
    "kind": "nvidia_dll",
    "path": "ABSOLUTE_PATH_TO_THE_SELECTED_DLL",
    "sha256": "LOWERCASE_WHOLE_DLL_SHA256",
    "model": "original"
  }
}
```

Omitting `model` preserves the original default. Explicit `original` normalizes to
the legacy serialized shape; non-default IDs survive serialization/helper transport.
`model_folder` and legacy `files` sources also carry model identity.
Disk manifests retain `dlss5-phoenix-model/1`; selected architecture and optional
manifest model ID must match. Existing containment, extent, checksum and layout checks remain.

```text
Selected model ID + container path/checksum
  -> registry lookup and implementation/handler preflight
  -> authenticate source and select the definition's named resource
  -> verify record count and layout
  -> selected decoder and selected graph
  -> selected execution backend
```

Windows resource reads use `LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE |
LOAD_LIBRARY_AS_IMAGE_RESOURCE`. Non-Windows reads parse PE data. Neither runs
NVIDIA DllMain/inference code. Resource positions may change: lookup is by name,
not hardcoded file offset. `small-fp8` must not silently fall back to `WEIGHTS_HT`.

Portable runtime startup preflights both handlers before backend loading. Replacement
validates the candidate before replacing the current model/graph/history. Status reports
`model_id`, `model_architecture`, `model_resource` and resource digest where available.
Portable capabilities expose the registry catalogue; implementation availability is not
a claim of downloaded weights, hardware support or vendor parity.

## Native Resource Compatibility Fix

Previously, generated native bundles gated on the entire old DLL checksum.
`PackedModel` now carries a hash of the actual selected resource. The bundle validates
that identity after authenticated loading, before accepting its packed weights.

- New production graph metadata includes `model_architecture` and
  `reference_model_resource` with name and raw-resource SHA-256.
- Older generated bundles resolve a known historical DLL checksum to the registry's
  reference-resource fingerprint, in memory. Their files/manifests are not rewritten.
- Unknown legacy identities retain exact-container-only support, not guessed equivalence.
- Disk-model editions retain their explicit compatible-weight path, authenticated by
  their own manifest/layout. They are not forced to equal vendor reference values.
- The picker verifies reference-resource identity and caches validation/rejection by
  path, metadata and model ID. It also retains known-container approval because its
  selected DLL can feed NVIDIA-code execution routes. Both verified DLLs are known.

**Resource equivalence is not approval to execute arbitrary DLL code or reuse
version-specific NVIDIA hooks.** The smaller model is implemented through portable
backends and native RTX; the latter currently has CPU plan/dispatch evidence only.
The source production graph now has additional identity metadata; old installed kernel
bundles are untouched. Packaging guards may require normal regeneration of a current
bundle. Never bypass stale-graph/manifest checks or rewrite hashes merely to force a pass.

## Validation Already Completed

`artifacts/current/model-selection-validation.json` records 11 passing compiled cases:
selection, legacy loading, resource identity, picker/cache and portable preflight on
both x86/x64, plus x64 native preflight. Tests include three/32-model registries,
missing handlers, both real DLLs, native bundle loading and all 41 kernel-file checksums.
Both DLLs produce identical original tensors and packed weights. Incorrect DLL hashes,
resource/architecture mismatches and in-memory weight mutations preserving layout are rejected.
Picker rejection caching and recovery also pass. Temporary builds were removed.

No GPU inference, games, visual comparisons or performance benchmarks ran for these
changes. Installed DLLs/configurations were not replaced; nothing was pushed or released.
Eight focused packaging Python tests also passed. Do not treat this as full-suite or
end-to-end rendering validation, or reuse unrelated earlier GPU receipts as proof.

To repeat the focused CPU checks from the repository root:

```powershell
python tools/test_model_selection.py `
  --model "../_dlls/nvngx_dlssnr.dll" `
  --model "../research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll" `
  --bundle artifacts/current/native-kernels `
  --ngx-include "../external/DLSS/include" `
  --output artifacts/current/model-selection-validation.json
```

Omit the external model/bundle/SDK options for asset-free selection checks. CTest
registers the portable selection, resource-identity and runtime-preflight contracts.

## Further Validation Checklist

1. Read `AGENTS.md`, this handoff, `docs/MULTI_MODEL.md`, and the current dirty diff.
2. Confirm the user's requested next scope. This handoff does not itself authorize a
   model port, GPU run, deployment, commit, push or release.
3. For a smaller-model port, inventory its real 77-record resource and independently
   verify shapes, packing, scaling, topology, shifts, resampling, skips and conditioning.
4. Implement its decoder/layout table and graph in its own folders. Keep the original
   checks and default unchanged; do not just lower a global count or flip `implemented`.
5. Audit shared helpers too: current packing/accessors, frame preparation, history,
   appearance/output profiles and repeat controls still contain original-model assumptions.
6. Add matching native bundles/contracts only where implemented. Portable handler support
   alone does not establish native RTX, x86 transport or every-backend compatibility.
7. Audit NVIDIA execution/preset selection separately. For example, the existing NVIDIA
   initialization in `src/integration/compatibility/wow64/model64.cpp` sets the render
   preset to zero. Do not assume a model ID is already mapped to an explicit vendor preset.
8. Upstream notes linked from the research README describe preset 1 as original and
   preset 2/default as smaller in 310.9.16. Treat those as reference leads, not local GPU
   verification. Check licenses before importing third-party implementation code.
9. Add stage-level, full-output, temporal and control regressions before advertising the
   smaller model. Request explicit authorization before GPU/reference/game tests.
10. Preserve both DLLs and all unrelated work. Keep `DO_NOT_TOUCH_DLSS_REFERENCE_20260907`
    read-only. Do not publish proprietary models, machine paths, private configs or logs.

Centralized versioned model storage, download/import UI, renderer-specific capabilities
and renderer-specific model-selection restrictions remain follow-up work. A shared Neural model
selector is now implemented for portable and native RTX execution. The native adapter
lives in `src/native/model/models/small_fp8/native_graph.h`; compact build/CPU receipts
are under `artifacts/current/small-model-support/native-rtx/`. The naming convention
in `models/README.md` is documentation; it has not moved any existing DLLs.


## Model-aware UI and explicit processing controls

The UI and processing follow-up is now implemented. See [PROCESSING_PIPELINE.md](PROCESSING_PIPELINE.md) for the behavior and capability boundaries. The older further-validation checklist above is historical; model/preset wiring and the smaller-model implementation are no longer unimplemented TODOs.

NVIDIA selection now maps Original to preset 1 and Lightweight FP8 to preset 2, with separate explicit DLL-default selection and artistic A/B/C Style. The authenticated selected DLL takes precedence over a game-local copy. Model profiles show the distinct graph/weight counts. Missing Lightweight resources and the Original-only experimental x86 driver-host ABI are rejected explicitly.

The optional explicit processing path adds insertion stages, current-frame Present guide reuse policies, 0–200% working scale, domains/filters/residual reconstruction, input/signal encodings, diffuse-white/exposure controls, UI/Hudless policies and 1–10 passes. Native refinement passes have independent histories. Existing profiles keep the legacy color bridge until this pipeline is enabled. Final-frame/feeder sources expose their actual timing and guide capabilities.

Focused x86/x64 CPU, shader compilation and real ImGui checks are recorded under `artifacts/current/processing`; development addons, dispatch libraries and helpers are compiled/linked there too. These are unembedded development candidates. No game, graphics device or GPU inference was run for these new controls, and no installed files or proprietary extracts were replaced. Earlier four-model inference parity applies to its tested recipe, not every new option combination.

The granular-control follow-up is also implemented for Lightweight: learned transformers 11–18, all eight 0–4 repetition controls, depth ceiling/floor, GPU-time target and actual game-FPS target on the direct timed route. Its native accounting uses 38 fixed launches plus five per retained repetition, versus Original's 114 plus five. Adaptive steps skip disabled tail blocks, and pending samples cannot enter a new observation window. Native stage-query storage now covers all ten processing passes. See the granular section of `PROCESSING_PIPELINE.md` and the focused CPU/UI receipts in `artifacts/current/granular-controls`; development binaries are refreshed in `artifacts/current/processing`. The millisecond budget is adaptive rather than a hard inference deadline. Route restrictions and the no-GPU-validation boundary above still apply.

Portable follow-up: both model graphs and granular scheduling now share `native_runtime/network_schedule.h` across Metal, Vulkan, HIP/AMD, SYCL and CUDA. The portable frame API adds optional depth/adaptive-budget metadata without changing its ABI. GPU budgets use actual backend GPU events across passes; FPS targets require caller application-present timestamps. The direct Windows addon consumes completed portable-worker GPU timing once per matching generation/epoch/model, and portable final-frame UIs enable GPU budgets only when the updated runtime reports support. See `PORTABLE_MODEL_CONTROLS.md` and `artifacts/current/portable-models`. CPU recording covers both models and backend feature profiles; Apple/HIP/SYCL compilation and backend GPU inference were not executed on this Windows host.
