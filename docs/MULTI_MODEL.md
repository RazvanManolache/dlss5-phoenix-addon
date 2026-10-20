# Model Selection and Architecture Boundaries

<!-- current-two-model-evidence -->
See [the current model/backend support matrix](SUPPORT.md) and [two-model benchmark](benchmarks/current/README.md) for the measured native/NVIDIA presets, portable devices, memory and schedule costs. Source availability and fresh device validation are reported separately.
<!-- /current-two-model-evidence -->

Phoenix DLSS5 separates **model identity**, **weight storage** and **execution backend**.
Native launch/group metadata and matched NVIDIA preset IDs now live in each
model definition. `NativeLayout` views that descriptor rather than selecting
between two hard-coded architectures. New architectures still need a verified
decoder, graph and compatible kernels; registry metadata alone is not support.
Both built-in models have portable decoders/graphs and native RTX launch support. See [Lightweight FP8 support](SMALL_FP8.md) for backend scope and validation.

The registry has no two-model limit. Model IDs and architecture IDs are strings, not
a two-value enum. The built-in definition table grows to any number of entries;
catalogues and validation iterate the registry rather than depending on slots or order.
Duplicate IDs and incomplete definitions are rejected. Multiple weight editions may
share one architecture's decoder/graph. A genuinely new architecture needs new
handlers, registered in the decoder and graph tables; an unknown handler never falls
back to the original model, even if its definition is incorrectly marked implemented.

| Model ID | Resource | Architecture | Phoenix implementation |
| --- | --- | --- | --- |
| `original` | `WEIGHTS_HT` | `phoenix-network37/1` | Existing decoder and graph |
| `small-fp8` | `WEIGHTS_CC_SMALL_FP8` | `phoenix-cc-small-fp8/1` | Portable decoder and 31-block graph; native RTX 78-launch schedule |

The smaller model's architecture ID is an internal Phoenix ID, not an NVIDIA API name.
DLL version is not model identity: NVIDIA DLSS5 310.9.16.0 contains both resources.

## Configuration

Add `model` to the existing `weight_source` object:

```json
{
  "weight_source": {
    "kind": "nvidia_dll",
    "path": "C:/Phoenix/Models/nvidia-dlss5/310.9.16.0/nvngx_dlssnr.dll",
    "sha256": "2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f",
    "model": "original"
  }
}
```

This selects the **old original model inside the new DLL**, not the smaller model.
Both the portable decoder and native bundle loader accept its original-model resource.
The DLL must match the configured whole-file checksum. Native compatibility then checks
the selected resource's name, architecture and exact reference hash, independently of
the DLL version or byte offset. This is resource-only Phoenix DLSS5 loading, not permission
to execute a different NVIDIA DLL's code through version-specific hooks.

- Omitted `model` means `original`; existing configurations retain their serialized shape.
- Explicit `original` normalizes to that same default representation.
- Non-default selections survive serialization, including helper configurations.
- `model` also applies to `model_folder` and legacy `files` sources. Unavailable models
  fail before loading; an explicit selection is never replaced by the original model.
- Unknown IDs fail, rather than being guessed from filename, version, size or record count.
- `small-fp8` selects its own 302-tensor decoder, 31-block portable graph and native
  RTX schedule. Source-built native bundles select the smaller graph in memory;
  legacy captured bundles remain original-only. No model fallback occurs.

Disk manifests keep schema `dlss5-phoenix-model/1`. Their `architecture` must match the
selection; optional manifest `model` must match too. Older manifests may omit `model`.
Digest, extent, layout and path-containment validation remains enforced.

## Shared NVIDIA DLL Location

The full Phoenix panel has a **NVIDIA DLL search folder** field and a folder browser,
available for Phoenix DLSS5 and NVIDIA DLSS5, in both x86 and x64 addons. This lets
multiple installations reference one large NVIDIA library without copying it into
each game directory. It does not download or redistribute the library.

The setting is saved in `ReShade.ini`:

```ini
[Phoenix]
NvidiaDllFolder=C:\Phoenix\Libraries\310.9.16
```

Or provide a deployment default in `Phoenix.json`:

```json
{
  "nvidia_dll_folder": "C:/Phoenix/Libraries/310.9.16"
}
```

These snippets are additions to an existing working configuration, not complete
standalone configurations. A saved panel setting takes precedence over the JSON
default. Clear the field to restore local-only discovery.

Search order is deterministic:

1. The existing configured/selected DLL path, or `nvngx_dlssnr.dll` beside the game
   executable when no DLL path was selected.
2. The same filename directly inside the configured search folder.

Relative filenames and search folders resolve against the game executable's
directory, not its current working directory. Helpers receive the resolved absolute
model path and an absolute search folder, so their private directory does not change
the selection. Versioned filenames are preserved; there is no recursive scan,
automatic newest-version choice or search through `PATH`.

Local files take precedence even when incompatible. Existing container, resource
and model checks still apply: an invalid local DLL produces an error rather than
silently falling through to a different shared version. Disk-model folder selections
remain separate; the DLL folder can still supply the NVIDIA comparison runtime.

Changing the folder is a resource selection, not a per-frame strength control. A
running NVIDIA DLL cannot be safely replaced in place: restart the game to load a
different library after one is already loaded. Model choices and supported backends
remain subject to the normal support matrix.

`python tools/test_dll_search.py --ngx-sdk <SDK headers> --imgui-source <matching ImGui sources>`
checks path precedence, settings/IPC round trips and the actual shared widgets in
both bitnesses, then compiles both production addon entry points without running
games or GPU inference.

## Source Organization

```text
src/native/model/
  definition.h                  Shared identity/architecture contract
  registry.h                    Lookup, availability and graph compatibility
  models/
    original/definition.h       Original resource/layout/tensor metadata
    small_fp8/definition.h      Smaller model identity and tensor contract
    small_fp8/native_graph.h    Smaller native RTX graph specialization
  weight_source.h               Authenticated source reading and packing
  portable_model_io.h           Bounds-checked named PE resource reader

native_runtime/
  model.h                       Common tensor access and decoder dispatch
  graph.h                       Shared operations and graph dispatch
  models/
    original/
      decoder.inc               Original tensor decoder
      weight_layouts.h          Original tensor layout table
      graph.inc                 Original graph recording
    common/decoder.inc         Shared packed-family decoding operations
    small_fp8/
      decoder.inc               Smaller-model decoder entry
      weight_layouts.h          Independent packed-family table
      graph.inc                 Smaller-model graph and shift schedule
  weight_layouts.h               Compatibility include for existing consumers
```

The `.inc` files are included inside the corresponding class. This preserves current
interfaces without adding a virtual interface to the inference path. Model-specific
graph recording is separate from shared backend operations. Some packing/accessor
helpers still encode the original layout: future decoders must verify their suitability.
Matching operator-family names alone does not establish compatible packing or semantics.

## Loading and Reporting

1. Resolve the stable ID from the shared registry.
2. Reject unavailable models before backend loading/device allocation in portable and native sessions.
3. Authenticate the DLL or disk resource without executing NVIDIA code.
4. Select the named resource and validate its record count and layout digest.
5. Dispatch its decoder and graph, without a default-to-original branch.
6. Report the actual ID, architecture, resource name and resource digest.

Portable and native status add `model_id`, `model_architecture` and `model_resource`.
Spatial-only native sessions report `none` rather than claiming to run a model.
`phoenix_runtime_capabilities` includes the registry's model catalogue, labels,
implementation availability and rejection reasons even without a session. Availability
here means compiled model support, not downloaded weights or compatible hardware.

Replacement validates a candidate before replacing the live model. Unsupported/unknown
replacements preserve the current graph/history. Successful replacements reset history
and refresh model status. Generated native bundles must match `model_architecture`;
older untagged source-built bundles authenticate their original graph before selecting
an architecture adapter in memory. Legacy captured bundles remain original-model only.

### Native Resource Identity

New production graphs include `model_architecture` and `reference_model_resource`
(`name` and raw-resource `sha256`). The older `supported_dll_sha256` remains for older
readers. A current reader loads and hashes the resource before declaring compatibility;
simply changing the configured DLL checksum is not sufficient.

Older generated bundles are migrated in memory: a known historical container checksum
resolves through the model registry to its reference resource fingerprint. No bundle
file, manifest, cached kernel or installed DLL is rewritten. Unknown legacy container
identities retain exact-container-only support; cross-container equivalence requires a
known reference fingerprint or explicit graph resource metadata. A mismatch never falls
back to another resource or model.

The add-on DLL picker performs the same reference-resource check and caches both valid
selections and rejection messages by path, file metadata and model ID. Invalid selections
cannot reuse a previous valid digest or trigger repeated full-file hashing on every draw.
Because a picked DLL can also feed NVIDIA execution routes, the picker still requires
a known container identity (now including the verified 310.9.16 DLL). An arbitrary DLL
with copied weights is not approved for code execution. Resource-only native loading
remains independent of container version once its supplied whole-file checksum is verified.

Compatible `model_folder` editions remain an explicit separate path: their own manifest
checksum, path containment and selected architecture/layout validate the supplied weights.
They are not silently restricted to NVIDIA's reference weight values. Legacy captured
bundles still enforce their existing full packed-allocation/record hashes.

## Adding an Architecture

1. Add its definition with exact resource/layout metadata, initially unavailable.
2. If its architecture is new, add decoder/layout/graph folders and register their handlers
   in the decoder and graph dispatch tables. Compatible weight editions reuse the existing handlers.
3. Verify preprocessing, conditioning, packing, topology, history and output rules.
4. Generate matching native bundles where applicable; do not reuse an unrelated bundle.
5. Add stage/output comparisons and original-model regressions before enabling support.
6. Bind UI controls to model/backend capabilities in a separate integration change.

The addon now exposes a Neural model selector, saved as ModelId and carried through
helper configuration. No automatic downloads or game deployment are performed. External DLLs stay where they are. See [model storage](../models/README.md).

## Focused Tests

```powershell
python tools/test_model_selection.py
```

Compiles/runs x86/x64 CPU-only selection, PE, legacy-loader and runtime-preflight
contracts; removes temporary builds. No weights required. Optional read-only fixtures:

```powershell
python tools/test_model_selection.py --model "C:/Models/310.8.0.0/nvngx_dlssnr.dll" `
  --model "C:/Models/310.9.16.0/nvngx_dlssnr.dll" `
  --bundle "C:/Phoenix/native-kernels" `
  --output artifacts/current/model-selection-validation.json
```

Optional fixtures assert the known original decoded-tensor digest in either DLL.
These are not GPU, visual or performance tests. CTest registers both asset-free
selection and runtime-preflight contracts.

Supply `--ngx-include C:/SDKs/DLSS/include` to also compile the x64 native library
and run its unsupported/unknown-model preflight. This check creates no D3D12 device.

Local validation includes selection, resource identity, DLL picker, legacy loading and
portable preflight on both x86/x64, plus x64 native preflight.
Both DLL fixtures produced the same 658-tensor catalogue as before the refactor.
The original graph-recording body was also checked unchanged after extraction.
The compact local receipt is `artifacts/current/model-selection-validation.json`;
external model paths/data are not included in the source package.

Registry regression fixtures cover three and 32 models, shared-architecture editions,
duplicate/incomplete definitions, order-independent lookup/catalogues and missing
architecture handlers. These fixtures are test-only; they do not advertise nonexistent
NVIDIA models or add unvalidated weights to the production catalogue.

With `--bundle`, both real DLL fixtures also pass full CPU-only native bundle loading,
resource authentication and all kernel-file checksum checks. Wrong DLL checksums, wrong
resource names/architectures and weight-byte mutations that preserve the original layout
are rejected. No NVIDIA DLL entry point or GPU kernel is executed.
