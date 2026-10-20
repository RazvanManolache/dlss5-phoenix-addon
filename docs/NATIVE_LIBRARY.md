# Native RTX and spatial library — Windows x64

<!-- current-two-model-evidence -->
For fresh Original and Lightweight GPU timings, memory and corresponding NVIDIA-preset parity, see [the current benchmark](benchmarks/current/README.md). The [support matrix](SUPPORT.md) distinguishes the standalone single-pass native C API from addon/portable multi-pass scheduling and the Original-only x86 driver-host experiment.
<!-- /current-two-model-evidence -->

`phoenix_native.dll` exposes the addon's optimized D3D12/NVAPI network executor
without ReShade. It also exposes the existing spatial resolution bridge, either
after inference or alone. This is a separate Windows extension; the portable
`phoenix.h` ABI and its backends remain compatible.

The current source supports both `original` and `small-fp8`. Select the smaller
model with `"model":"small-fp8"` inside `weight_source`, using the 310.9.16 DLL and
a source-built kernel bundle. Its model-specific schedule has 78 neural launches;
the original has 154. Bundle files and kernel binaries are shared and remain
authenticated. Legacy captured-plan bundles support only the original model.
The smaller native path has build/CPU checks and a subsequent matched GPU comparison
at 720p, 1080p and 4K, with byte-identical NVIDIA reset/history captures on those
fixtures. General parity remains unverified. See [Lightweight FP8](SMALL_FP8.md).

## Build

Requirements: Windows x64, MSVC C++20, Windows SDK, CMake 3.22+, Python 3.10+
for build tooling, and the external NGX SDK headers. See the links and pinned
dependencies in [the build instructions](../README.md#build-from-source) and
[external dependencies](../third_party/EXTERNAL_DEPENDENCIES.md).

```powershell
cmake -S native_runtime -B build/native-library `
  -DPHOENIX_ENABLE_NATIVE_RTX=ON `
  -DPHOENIX_NGX_INCLUDE="C:/SDKs/ngx/include"
cmake --build build/native-library --config Release --parallel 4
ctest --test-dir build/native-library -C Release --output-on-failure
cmake --install build/native-library --config Release --prefix build/native-sdk
```

The installed CMake package exports `Phoenix::Native`, the import library,
DLL and [C header](../native_runtime/phoenix_native.h). Consumers need the
Windows SDK, but no NGX headers. Link `Phoenix::Native` and place the DLL beside
your executable. Microsoft Visual C++ x64 runtime is required.

For inference, also build the production kernel bundle:

```powershell
python tools/production_build_native_kernels.py native_kernels `
  --compiler "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe"
```

The bundle is generated from the repository's sources. The current production
catalogue targets `sm_120` and was validated on RTX 5080; this is not a claim of
support for every RTX generation. Spatial-only execution needs neither this
bundle nor a model nor NVAPI kernel support. It was also tested on D3D12 WARP.

## Configure a session

Pass UTF-8 JSON to `phoenix_native_create`. Example: run the model at 720p and
spatially enlarge its output to 1080p:

```json
{
  "width": 1280,
  "height": 720,
  "output_width": 1920,
  "output_height": 1080,
  "bundle_path": "C:/Phoenix/native_kernels",
  "weight_source": {
    "kind": "nvidia_dll",
    "path": "C:/Models/nvngx_dlssnr.dll",
    "sha256": "SHA256_OF_YOUR_SUPPORTED_DLL"
  },
  "repeats": [1,1,1,1,1,1,1,1]
}
```

- `width/height`: source dimensions, 1..16384. `output_width/output_height`
  default to these; select a different output size for up/downscaling.
- `neural_width/neural_height`: inference canvas, defaulting to source size.
  The source is fitted into this canvas with centered padding, so choosing a
  different processing aspect does not stretch the model input. Output samples
  exclude padding. The final output fills the requested dimensions; keep the
  source aspect if you want no display stretch.
- `repeats`: eight immutable block repetition counts, each 0..4. Defaults are
  all one. Changes require a new session and reset history.
- `weight_source` uses the existing authenticated DLL/disk loader. For disk
  models use `{"kind":"model_folder","path":"C:/Models/compatible-model"}`.
  No model or NVIDIA DLSS5 DLL is included in the library. The NR DLL is mapped as
  resource data; NVIDIA's NR evaluation code is not executed.
- `adapter_index`: optional DXGI adapter index when the library creates its
  device. Without it, native inference selects an NVIDIA adapter. Unsupported
  devices fail explicitly. Supply both D3D12 device and DIRECT queue to use an
  application's existing device instead.
- `spatial_only: true`: skip model loading and NVIDIA DLSS5 kernel initialization.
  `bundle_path` and `weight_source` are unnecessary. `warp: true` explicitly
  selects the software device for this mode only.

For spatial-only 640×360 → 1920×1080:

```json
{"width":640,"height":360,"output_width":1920,"output_height":1080,"spatial_only":true}
```

## Texture and host-buffer calls

```c
PhoenixNativeControls controls;
phoenix_native_default_controls(&controls);
controls.reset = 1; /* First frame, scene cut, or discarded temporal history. */

PhoenixNativeSession *session = phoenix_native_create(config_json, NULL, NULL);
/* Check session, then either supply D3D12 textures or half-float host planes. */
int result = phoenix_native_frame_host(
    session, input_rgba16f, input_bytes, input_row_bytes,
    motion_rg16f, motion_bytes, motion_row_bytes,
    output_rgba16f, output_bytes, output_row_bytes, &controls);
/* Copy phoenix_native_error() immediately if result is -1. */
/* Set reset=0 for subsequent contiguous frames with valid motion. */
phoenix_native_destroy(session);
```

All functions using a session run on its creator thread. The texture function
accepts RGBA16F color/output and optional RG16F motion, together with their
current resource states. It validates device, dimensions, format and aliasing,
records into its own command list, submits on the selected DIRECT queue, waits
for completion and restores resource states. The caller orders previous image
producers on that queue and serializes queue submissions during the call.
Cross-queue producers need a caller-managed fence wait first.

The host function accepts IEEE half bit patterns with explicit byte capacities
and row strides. It uploads input, uses the same native texture path, and reads
back output. These transfers cost time; applications with GPU images should use
the texture call. Inputs use the native model's SDR proxy domain, not arbitrary
linear HDR. Color conversion/tonemapping is the application's responsibility.

Motion is optional; missing motion disables temporal reuse. Motion uses source
pixel displacement, multiplied by the supplied motion scales. Spatial reduction
adjusts the vectors to inference pixels. Alpha comes from the original source,
resampled independently when output dimensions differ.

The original V1 controls and output path remain compatible. The candidate's
three-frame V1 output matched the unchanged SDK exactly. New experimental
appearance calls are `phoenix_native_frame_d3d12_v2` and
`phoenix_native_frame_host_v2`; the latter consumes the bounded `phoenix_frame_v3`
structure. They support the [declared Phoenix map/style/UI/field contract](LIBRARY.md#versioned-host-planes-and-appearance-contract).
The standalone native network executes one pass and owns its native temporal
and noise sequence. Host RGB32F is converted through native half textures;
zero-intensity regions may therefore differ from source float32 by half
quantization (maximum 0.000244141 in the measured resize fixture).

For D3D12 V2, initialize `struct_bytes`, `version=2`, default native controls,
`intensity=1` and zero absent resource descriptors. Color/output are RGBA16F,
motion RG16F, map/UI/field RGBA32F, alpha R32F, backbuffer RGBA16F. Each plane
includes its observed state, restored before synchronous return. UI and UIAlpha
are alternatives; all resources must be distinct and on the session device.
Source, neural and output extents must agree for the advanced path. The native
host adapter currently refuses explicit depth/validity planes. Arbitrary HDR UI
and scaled optional planes are not supported by this new contract.

The GPU map is bound to the retained preparation kernel's existing descriptor
at byte 32; its G/B affect conditioning. Display grade/intensity/R/UI run after
recurrent history, and field handling uses a separate canonicalization stage.
Mask texture allocation is lazy in the standalone library. No NVIDIA model DLL
code is executed by these calls.

`phoenix_native_reconfigure(&session, config_json)` creates a replacement on the
same device/queue before retiring the old one. It resets history, can resize or
select compatible replacement weights, and leaves the old pointer unchanged
on failure. GPU tests covered an invalid resize and a successful 320�320 resize
with a zero-intensity map. Borrowed devices still require queue serialization;
reconfiguration temporarily holds both sessions' resources.

`phoenix_native_device` returns AddRef'd interfaces for applications using the
owned-device path; release them normally. `phoenix_native_status` returns JSON,
including actual path, geometry, GPU processing time, complete-call time and
the owned allocation ledger. Query the required size first. GPU processing
excludes input preparation and host transfers; it includes inference and output
copy/resize. Allocation accounting is not total process RAM/VRAM.

Destroy drains outstanding library work before releasing resources. On an
unresolved fence failure it returns -1 and retains the handle; do not free it
yourself. A failed GPU session rejects further frames. Recreation provides
resize/reconfiguration; no silent backend fallback is performed.

## Validation

See [the 15 October checkpoint](NATIVE_LIBRARY_VALIDATION_2026-10-15.md).
The standalone DLL, C header, software spatial tests and portable ABI tests
build together. `native_library_probe` exercises the public API on real hardware:

```powershell
build/native-library/Release/native_library_probe.exe config.json capture/input results.json gpu 128
```

The capture prefix uses the `.proxy.bin` RGBA16F and `.motion.bin` RG16F files
generated by `tools/prepare_native_fixture.py`. Use `host` to exercise transfers.
Builds and probes do not install an addon or change a game.
