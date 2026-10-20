# Lightweight FP8 model support

<!-- current-two-model-evidence -->
Fresh GPU execution now covers both models on native RTX, CUDA, NVIDIA Vulkan and AMD Vulkan, alongside CPU. See [the current benchmark](benchmarks/current/README.md) for timings, memory, per-block/pass costs and exact validation scope, and [the support matrix](SUPPORT.md) for unavailable platforms. Metal/HIP/SYCL remain unmeasured in this refresh.
<!-- /current-two-model-evidence -->

Phoenix can load and execute the `WEIGHTS_CC_SMALL_FP8` model from NVIDIA DLSS5
310.9.16.0 through its portable runtime and the Windows x64 native RTX executor. The stable model ID is `small-fp8`.
The original model remains the default, and its decoder fingerprint is unchanged.

Select **Lightweight FP8** in the addon's **Neural model** menu, choose the
310.9.16 DLL as the model source, and select **Native RTX** or a portable processing device. The
model choice is saved as `ModelId` and carried through helper configurations.
Changing it rebuilds the session and resets temporal history.

For a library client, add the model ID to the existing source configuration:

```json
{
  "weight_source": {
    "kind": "nvidia_dll",
    "path": "C:/Phoenix/Models/310.9.16/nvngx_dlssnr.dll",
    "sha256": "2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f",
    "model": "small-fp8"
  }
}
```

Compatible disk-model folders can also specify this model, with architecture
`phoenix-cc-small-fp8/1`. Selecting the smaller model from a DLL without its resource
fails; it never substitutes the original resource.

## Implementation

The new architecture owns its decoder entry, packed-family table and graph under
`native_runtime/models/small_fp8/`. The common byte-decoding operations are shared
with the original model, while each architecture supplies its own table and tensor
count. The smaller resource contains 77 serialized records, decoded to 302 tensor
views. Its graph records 31 learned blocks with eight global transformers.

The FP8 graph recorder is organized into named encoder, transformer, decoder and
output-head helpers, with explicit stage/skip tables and documented shift units.
The native graph adapter likewise separates launch selection, edge and weight
remapping, ABI shift patches and workspace lifetimes. This is a readability
refactor: complete native graph and portable command/weight fingerprints match
their pre-change values, and the CPU reset/history/control outputs are unchanged.
See the [implementation map](../native_runtime/models/small_fp8/README.md) and
`artifacts/current/small-model-support/refactor/` for the focused receipts.

The host constructor in the inspected DLL selects stage counts `{2,2,2,2,2}`.
Its decoder shift helper at RVA `0x568f0` uses `(stage_count + block_index) % 4`;
therefore every two-block decoder level uses phases 2 and 3. The encoder uses
phases 0 and 1. This includes the 512-channel split-window levels. The smaller
graph has its own block numbering, skip connections and eight repeat controls.

The model's history blend scalar is loaded from block 30, rather than block 70.
Its value is `0.76708984375`. Style/control descriptor data at offsets
`+0x20..+0x290` is byte-identical between both 310.9.16 model descriptors, allowing
the existing recovered style table for these two exact reference resources.
Custom weight editions retain the existing rejection for unverified style data.

Portable operator backends share this model graph: CPU, CUDA, Vulkan, HIP, SYCL and
Metal. Backend availability still depends on the built provider and target device.
Fresh device execution and exact-comparison boundaries are listed in the current benchmark linked above; shared graph support alone does not establish parity on untested hardware.

The optimized native RTX executor now has a separate 78-launch schedule for this
model, using all 77 packed weight records and the existing 41 authenticated kernel
modules. The original schedule retains its 154 launches. The smaller schedule
reconnects feature/publication edges, remaps encoder skips and history gain, adjusts
shifted grids and decoder phases, and recomputes workspace lifetimes and barriers.
Depth reduction, block repeats, control patching and ablation groups use the selected
schedule. Eight transformer blocks remain available in both models.

The architecture adapter is `src/native/model/models/small_fp8/native_graph.h`.
It specializes an authenticated source-built bundle graph in memory, after verifying
its file hashes. It does not rewrite the bundle, remove integrity checks or duplicate
kernel binaries. Legacy captured-plan bundles remain original-model only. Existing
NVIDIA hooks and vendor preset execution routes are unchanged.

For `phoenix_native_create`, use the same `weight_source` shown above plus your
normal absolute `bundle_path` pointing to a source-built native kernel bundle.
The local library/config example and updated addon are under
`artifacts/current/small-model-support/native-rtx/`. The native API reads the selected
DLL resource as data. Select the model explicitly; it never falls back to original.
The current kernel bundle targets `sm_120`; model support does not extend hardware
coverage to other RTX generations.

## Validation

A subsequent explicitly requested [four-way GPU comparison](../artifacts/current/four-model-comparison/README.md)
ran both Phoenix production native graphs and both NVIDIA presets from 310.9.16,
at 720p, 1080p and 4K on RTX 5080. All runs produced finite output and repeatable
resets; the initial debug smoke reported no D3D12 errors. NVIDIA launch observation
confirmed preset 1/original (156 launches) and preset 2/small (80 launches).
The corrected comparison explicitly enables automatic masking and disables inverted
depth on both routes. Full RGBA16F hashes match byte-for-byte for both models at
all three resolutions, for reset, history and repeated-reset captures. The earlier
reported RGB errors were caused by the harness leaving automatic masking disabled
only on Phoenix; those runs and conclusions were replaced. No production model
code needed a change. At 1080p, corrected median GPU times are 3.869/2.298 ms for
Phoenix original/small and 3.799/2.221 ms for NVIDIA original/small. These timings
exclude uploads/readback and CPU gaps; they are not game FPS. A static zero-motion
museum fixture supports these findings; moving-game quality, HDR and general
parity remain unverified. The earlier CPU-only receipts retain their original scope.


`native-rtx/validation.json` records CPU-only native checks with the actual DLL and
model-free bundle: resource identity and all 77 weight bindings, feature/publication
connections, shifted grids, live memory ranges, 12 resolutions including odd sizes
and 4K, immutable control patching, individual/batched recording, all eight depth
settings, all 15 ablation groups, and repeat synchronization metadata. Original graph
selection and launch packets are checked separately. The native DLL and addon are
built; these checks do not execute kernels on a GPU.

Repeat this native check/build with:

```powershell
python tools/test_native_small_model.py `
  --model ../research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll `
  --bundle artifacts/current/native-kernels --ngx-include ../external/DLSS/include `
  --build build/native_small_model `
  --output artifacts/current/small-model-support/native-rtx/validation.json --frontend
```

The earlier portable execution receipts remain below.

`artifacts/current/small-model-support/` contains compact local receipts:

- `selection-validation.json`: x86/x64 selection, authentication, identity, original
  tensor regression, smaller tensor decoding, graph construction, and native-route
  preflight. Both original DLL containers remain valid for the original model.
- `graph-validation.json`: real small-resource fingerprint, all 31 block IDs,
  decoder phases, output extents and allocation release at two padded sizes.
- `cpu-inference.json`: actual public runtime API frames with finite nontrivial
  output, deterministic reset, temporal-history use, style/control changes and
  original/small model replacement. Invalid replacement retains the live session.
- `style-table-evidence.json`: exact static descriptor comparison.
- `shared-panel-validation.json`: both UI bitnesses, including real dropdown
  interaction selecting and restoring the new model.
- `model-validator.json`: the actual Validate model task in both frontend
  bitnesses, each checking 77 records and executing three CPU frames via the
  runtime broker.

The local model-free candidate is under
`artifacts/current/small-model-support/runtime/`. It includes the new x64 addon,
runtime, CPU provider, broker and standalone client; CUDA/Vulkan providers are
reused unchanged from the existing local candidate and were not GPU-tested here.
Its `phoenix_native_cpp.json` selects the supplied external DLL and `small-fp8`,
with CPU as the default. Nothing was installed into a game. The small
`smoke-job.json` alongside it can be run with that candidate's `phoenix_frame.exe`
and a fresh output filename.

The original 658-tensor catalogue stays
`db9949368cd15e17886ec5f134e117c1de52889db6db6e4dd59f456bd702743e`.
The smaller 302-tensor catalogue is
`cb1269466e5494f0aae678f3eb7801655efde2e065d1febe6e2ec5c3f638dd3f`.

Repeat the focused CPU checks with:

```powershell
python tools/test_model_selection.py --model ../_dlls/nvngx_dlssnr.dll `
  --model ../research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll `
  --small-model ../research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll
python tools/validate_small_model.py --runtime PATH_TO_BUILT_RUNTIME `
  --model PATH_TO_310_9_16_DLL --output PATH_TO_RECEIPT
```

No NVIDIA reference inference, GPU inference, game, device deployment, commit,
push or release was performed for these checks. Finite CPU output and lossless
tensor decoding do not certify numerical or visual equivalence to NVIDIA.
