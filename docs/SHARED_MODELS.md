# Shared native model library

DLSS Everything can download optional full-resolution model exports into its
component library. The installer writes that library's absolute path into each
game's `Phoenix.json` (`model_library_directory`) and ReShade settings
(`Phoenix.ModelLibraryDirectory`). The model weights stay centralized.

Phoenix's **Shared native models** picker appears only when the configured
library contains available, supported packages. Selecting an entry sets the
existing model-folder source and matching architecture together. The usual
model reload/restart policy applies. New downloads are discovered within two
seconds while the panel is open. Missing entries do not replace the active source.

The initial optional packages are bdsqlsz's native-precision canonical exports
of **Original** and **Lightweight FP8**, converted by DLSS Everything's C# importer
into Phoenix's packed model-folder format. All 153 Original and 77 Lightweight
payload records were independently compared byte for byte with the NVIDIA
310.9.16 resource payloads. The importer also verifies every published stage hash.
The resource envelope is regenerated, so its file hash differs from the DLL's
resource hash; the actual weights and auxiliary payload bytes are identical.

These are alternate sources for the existing two networks, not newly trained
models. They use Phoenix's existing C++ engines and architecture-specific controls,
including the eight central blocks, repeats, passes and supported adaptive budgets.
They add no fixed image-size restriction, tiled inference, Python dependency,
PyTorch runtime, ONNX runtime or external wrapper process. Existing backend and
hardware limitations still apply.

The fixed-256 ONNX Reference/FP16/INT8 exports are excluded. Duplicate mirrors
and NVIDIA-DLL wrappers are not advertised as additional models. This change
does not claim a native port of SD.Next's different arithmetic or AMD async reuse.

The shared index is small and contains only relative package paths:

```json
{
  "schema": "phoenix-model-library/1",
  "models": [
    {
      "id": "canonical-original",
      "label": "Original native export",
      "model": "original",
      "manifest": "models/canonical-original-REVISION/model.json",
      "manifest_sha256": "LOWERCASE_SHA256_OF_MODEL_JSON"
    }
  ]
}
```

Discovery checks bounded metadata, known architecture, manifest checksum,
contained relative paths, and resource presence/size. Full resource checksums
and native record-layout checks occur again when the existing model loader opens
the source. No large model is hashed every overlay frame. Package directories
are immutable while games may hold mappings; a repaired download publishes a new
directory and atomically switches the index after verification.

Validation receipts: `artifacts/current/model-library-compile-tests.json` covers
x86/x64 discovery, authenticated native packing of both real imports, shared
widget regression checks and both production addon translation units. Those
checks use CPU only.

The [20 October benchmark refresh](benchmarks/current/README.md#dll-weights-versus-shared-model-exports)
additionally ran 48 isolated inference cases from the real canonical exports,
with two fresh processes per model, route and extent. Native RTX covers 320x180,
1080p and 4K; CPU, CUDA and NVIDIA/AMD Vulkan cover 320x180 and 1080p, with the
single-thread CPU reference limited to 320x180. Every export's reset/history
captures match its DLL-loaded baseline exactly. The matched 1080p network-buffer
allocation totals are also unchanged across all tested routes.

Native 1080p session creation measured approximately 697 to 487 ms for Original
and 630 to 378 ms for Lightweight (DLL versus export). These are isolated setup
boundaries, not application launch time or controlled cold-cache trials. Native
creation excludes the probe's initial whole-DLL digest; startup peak RAM also
includes probe-only allocations that production mapped hashing does not require.
Steady timing, process ranges, RAM/VRAM and source identities are in the report.
No games, VR compositors or x86 transport routes were benchmarked in this refresh.
