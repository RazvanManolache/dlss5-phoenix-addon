# Windows CUDA integration preview

The Linux integration's shared CUDA graph cache and whole-frame GPU profiling
also build on Windows. The cache retains two command schedules, which avoids
rebuilding a graph when two stable schedules alternate. Profiling can measure
one complete dispatch span instead of placing events around every operation.
These changes preserve the separately selectable portable CUDA backend.

Windows can additionally build the native CUDA executor introduced by the Linux
work. This is an optional x64, SM120 executor for encoded SDR proxy input.
Original uses 154 neural launches and Lightweight FP8 uses 78. It uses the same
authenticated external model loader and production kernel sources, with CUDA
texture/surface objects and graphs. Model weights are not included.

## Build

With MSVC, CMake, Ninja and CUDA Toolkit 13.1 available:

```powershell
python tools/build_windows_cuda.py --cuda "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" --ngx-sdk C:\path\to\DLSS\include
```

The builder compiles in a temporary directory and promotes the runtime DLLs,
standalone client, model-free kernels, headers and CUDA license to
`artifacts/current/windows-cuda`. `BUILD.json` records source and output hashes.
A successful build is compilation evidence; GPU tests are separate.
Add `--vulkan` to include the source-built Vulkan provider and SPIR-V in the
same current runtime/package. `--vulkan-headers` and `--glslang` accept external
dependency locations. Windows x86 CPU builds use the dispatched AVX2/AVX-512
provider; other platforms retain the portable reference provider.
The current recipe permits the installed MSVC version through nvcc's compiler
version gate. Runtime validation remains necessary for that toolchain pairing.

After the addon bridge build below passes, `python tools/package_windows_cuda.py`
assembles the source-verified addon pair and CUDA runtime in
`artifacts/current/windows-linux-integration`, with an installer-compatible runtime
configuration and complete SHA-256 inventory. This creates a local preview;
it does not publish a release. SDK headers stay outside the runtime directory.

## Select native CUDA in the addon

Use a matching rebuilt x64 addon. In **Graphics compatibility / Portable**, select
the absolute runtime folder, explicitly allow an unvalidated device, and use:

```ini
[Phoenix.Compatibility]
ExecutionMode=portable
PortableRuntime=C:\absolute\path\to\windows-cuda
PortableDevice=native-cuda:REPLACE_WITH_32_LOWERCASE_HEXADECIMAL_UUID_DIGITS
PortableAllowUnvalidated=1
PortableFallback=0

[Phoenix]
MotionEstimation=1
RefinementPasses=1
Experimental=0
ProcessingChain=0
```

Select an external Original or Lightweight FP8 model through the existing model
picker. The native CUDA runtime now retains GPU history, consumes motion/depth
and per-pixel validity, and accepts frame-level tone, structure, skin, masking,
intensity, style, block depth/repetition, and whole-network pass controls.
Bounded V3 adds UI, coverage, backbuffer, control-mask and distortion planes.
Style presets require a model edition with an authenticated preset table.

On Windows, the shared native pipeline SDK supplies estimated motion when engine
vectors are unavailable, scene detection, monitoring, latency policy, appearance,
processing resolution, scaling/color/fusion stages, and per-stage histories.
Those stages surround the selected CUDA neural executor; the rendering GPU and
CUDA device remain explicit. Model changes reset histories and replace immutable
weights/modules; rejected frame validation does not advance history.

The SDK's neural leaves explicitly disable their own motion/scene services and
consume the parent's guides. Private D3D12 devices use ReShade's published unwrap
interface when available, retaining the creator proxy through child retirement.
This avoids recursively processing private queues and mixing proxy/native objects.
Loading clears use ReShade's descriptor and command APIs so its immediate list
also submits frames that contain only the loading gate.

CUDA sessions retain and explicitly bind their primary context for evaluation
and destruction, restoring the caller's exact context on return. Selecting a
device ordinal alone did not reliably isolate evaluation from optical-flow
context changes in the DX11 host. Arena clears and uploads share the session's
nonblocking stream so initialization cannot race a weight upload.
History commits also run on that stream and retire before the host frame call
returns, so a following frame cannot sample a pending default-stream copy.
The D3D12 path combines input/output validation into one scalar readback. Invalid
inputs are sanitized before speculative execution in private scratch; rejection
publishes neither history nor caller output. After validation, one conditional
kernel preserves raw half history and composes presentation; invalid frames skip
both writes. The external semaphore signal precedes the same validation readback,
so the completed transaction needs no second CUDA wait before D3D12 publication.
Unchanged boundary arguments reuse their graph parameters, and timing events are retained
as explicit graph nodes rather than submitted separately on every frame.
Returning from a skipped/repeated workload to all eight blocks restores the
original graph, removing the residual-preservation copies that adjusted arena
lifetimes require. `retained_residual_copies` exposes this in diagnostic status.

The packed byte API remains a fast reset-only spatial contract. The addon selects
it only for a full-network, one-pass frame with optional services disabled.
Temporal and extended requests use the corresponding full frame/pipeline path;
they are never silently reduced to reset-only processing. The runtime stays
opt-in and does not replace the default Direct3D executor or broaden SM120/SDR
support. NVIDIA comparison sources still use the existing NVIDIA route.

The standalone C API uses `backend: "native-cuda"` and
`device: "cuda:<32 lowercase hexadecimal UUID digits>"`. The extra `native-`
prefix is the addon's device selector, which distinguishes this executor from
portable CUDA. See [the native executor guide](LINUX_NATIVE_CUDA.md) for external
model configuration. Bounded V2/V3 calls accept temporal metadata; the packed
V1 extension deliberately retains its reset-only contract.

## Transport and measurements

[phoenix_rgba8.h](../native_runtime/phoenix_rgba8.h) exposes the platform-neutral
`phoenix_runtime_frame_rgba8` entry point. The old Linux symbol remains an alias.
The API checks capacity, extent, row stride, format and reset flags. Conversion
runs on CUDA; alpha and row padding survive, and rejected calls leave caller
pixels unchanged. Native create, frame and destroy calls restore the caller's
exact CUDA context, including a pre-existing non-primary context.

The Direct3D compatibility route still reads pixels to host memory and uploads
the result at its outer capture/presentation boundary. Inside the shared Windows
SDK, native CUDA neural stages now use D3D12-CUDA shared textures when the SDK's
DIRECT queue and selected CUDA device have the same adapter LUID. Input/output
copies, half/float conversion, motion scaling, appearance and history remain on
the GPU. A shared D3D12 fence orders ownership between the two APIs. Small scalar
validation readbacks remain; neural pixel readback/upload does not.

[phoenix_cuda_d3d12.h](../native_runtime/phoenix_cuda_d3d12.h) documents the optional
version 1 synchronous texture entry point. It borrows RGBA16F color/output and
optional RG16F motion, restores resource states and the caller's CUDA context,
and retains the queue/device until session retirement. One network pass is
accepted per leaf; the shared SDK still owns processing stages and their histories.
Changing models retires imported textures before destroying their CUDA stream.
This follows NVIDIA's [CUDA 13.1 D3D12 interoperability contracts](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/graphics-interop.html).

A different rendering adapter, an older runtime without this optional symbol, or
explicit `d3d12_interop: false` uses the existing host transport. Invalid GPU
requests and evaluation failures are reported rather than silently retried.
Host fallback reuses session buffers, and Windows half conversions use F16C only
when the CPU and OS support it. The SDK also skips identity resampling/copies
where geometry, resource state and lifetime permit. Actual scaling and sharpening
retain their existing processing paths.

`pixel_transport` reports the selected leaf transport and
`neural_host_transfer_bytes` counts neural pixel traffic per frame. Zero here
describes the shared-texture leaf, not the entire game capture/presentation path.
Packed transport removes the full RGB32F CPU conversion and transfer
from the native runtime boundary; it is not zero-copy. Readback telemetry now
separates CPU map/copy time (`last_cpu_readback_ms`) and logical frame bytes from
capture, upload and composition. Direct3D chooses its readback memory; Vulkan
host-cached allocation flags are not applicable to Direct3D heaps.
DX11 explicitly flushes its immediate command batch after signaling a capture
fence. This submits work and does not wait for GPU completion; the integrated-GPU
DX11 validation still showed slow fence retirement after this flush.

GPU `operator_ms`, synchronous host API `call_ms`, and live presented FPS measure
different scopes. Whole-frame profiling also changes event scope relative to
older operation-summed timing. Compare synchronous call times for paired builds
and keep capture/presentation costs separate.

## Validation

The paired validation uses the separate `dlss5-everything-testing` project:

- Preserve a clean deterministic museum capture before changing the runtime.
- Run `tools/profile_linux.py` against the old and new portable CUDA DLLs on
  that identical capture. The script also runs on Windows and can save PNGs.
- Run native float V2 and packed RGBA8 on the same image and compare output.
- Run `tests/linux/native_cuda_contracts.py --check-context` for both models.
- Run the testing project's `tools/run_cuda_live.py` with the rebuilt addon,
  explicit runtime/device and selected Direct3D fixtures. It uses the offline
  installer and authenticated application controls, and removes its staging.
- Compile and run x86/x64 bridge, policy, availability and IPC tests with
  `python tools/validate_portable_bridge.py --build-addons --ngx-sdk PATH`.

Before/after receipts live in the testing project's explicitly requested
`runs/windows-linux-before` and `runs/windows-linux-after` directories. Those
local receipts distinguish native numerical differences and existing automatic
route failures from successful contract or fixed-image tests. They do not
establish game-wide compatibility, perceptual parity or temporal quality.

### Reset-only baseline, 2026-11-09

On an RTX 5080, the final runtime passed 23 native contracts for each model,
including exact caller-context restoration, rejected-input atomicity, byte-plane
padding/alpha, and standalone-client equivalence. Windows x86/x64 software
contracts passed 854/852 checks. Four Linux C++ software contracts, five launcher
tests and C-header compilation also passed under WSL; no Linux GPU inference
was performed in this Windows validation.

For the frozen 1280x720 Lightweight FP8 input, three warmup calls followed by
twelve measured calls produced these synchronous host API medians:

| Path | Median call time |
| --- | ---: |
| Portable CUDA before | 31.470 ms |
| Portable CUDA after | 35.930 ms |
| Native CUDA, float V2 | 12.976 ms |
| Native CUDA, packed RGBA8 | 2.398 ms |

Portable before/after PNG pixels were byte-identical. Native float/packed PNG
pixels were also byte-identical. Native versus portable differed by 0.4285 RGB
code values on average, with a maximum of 10. The portable path showed no speed
improvement in this short run; clocks and competing desktop work were unmanaged.

The final packaged native runtime passed the DX12 fixture's conditioning,
unsupported-motion rejection/recovery, fresh moving output, minimize/restore,
and odd/square resizing checks. DX11 presented enhancement and conditioning
changes but failed fresh-output acceptance due to slow capture-fence retirement
on the integrated rendering GPU. Automatic routing reproduced the baseline's
NVAPI failure and DX12 device error. These failing routes remain unresolved.

### Temporal and extended feature validation

Current follow-up receipts are under `runs/windows-linux-after/features` in the
testing project. `tests/linux/native_cuda_features.py` exercises motion/history,
validation-failure atomicity, depth/validity, intensity, V3 UI/distortion/masking,
truncated/bypassed/repeated blocks, multiple passes, adaptive GPU budgets and
model replacement for both models. Modified schedules preserve retained features
outside the original tensor liveness intervals, including the all-blocks-bypassed
case; this prevents an aliased decoder repack from overwriting its input.
The contract suite also changes the caller to another same-device CUDA context
after session creation and checks both output and exact context restoration.
The final package passed 24 API contracts and 13 GPU feature checks for each
model, plus shared-SDK conditioning tests with and without optical flow. The
x86/x64 addon software suites passed 857/855 checks. Linux software contracts,
launcher tests and final native-CUDA source compilation passed under WSL;
Linux GPU inference was not run.

Both DX11 and DX12 live feature runs passed with the rendering adapter explicitly
set to the RTX 5080. Each run used the real installer and application control
addon, checked visible conditioning/style/intensity and chain changes, persistent
native history with optical flow, workload changes, fresh moving output, memory
growth, minimize/restore and odd/square resizing, then restored the fixture.
Both also switched to the packed spatial path and back to the temporal SDK.
This follow-up does not qualify Automatic routing or other graphics APIs.
Those earlier temporal receipts used host transfers through the shared SDK.
The shared-texture follow-up below measures the optimized temporal path; the
packed spatial timing above is not a full-pipeline performance result.

Use `tools/run_cuda_live.py --features` in the testing project for actual addon
presentation, application controls, estimated-motion history, appearance,
workload/chain changes and resizing. The receipts distinguish kernel/API checks,
live feature behavior, and existing rendering-route failures. The reset-only
baseline numbers above do not measure the full temporal pipeline.

### Shared-texture temporal optimization, 2026-11-09

The receipts under `runs/windows-linux-after/optimization` compare the accepted
host-transfer build with the shared-texture build on the same fixed 1280x720
face capture and RTX 5080. Each model uses three paired trials, five warmup
frames per trial and sixty measured calls per build. Median times below cover
the entire synchronous `phoenix_native_frame_host` call, including optical flow,
neural execution, appearance and host input/output. Game capture/presentation is
outside this measurement; these numbers are not game FPS. Desktop GPU clocks
were unmanaged.

| Model | Before | After | Ratio |
| --- | ---: | ---: | ---: |
| Lightweight FP8 | 67.546 ms | 5.476 ms | 12.33x |
| Original | 86.130 ms | 7.437 ms | 11.58x |

Every measured frame, initial reset output and final temporal output matched the
old build exactly. Forced host fallback also matched. Both model editions pass
24 native API checks, 13 feature checks, 11 D3D12-CUDA transport contract groups,
and SDK conditioning with/without estimated motion. Transport checks include
nonunit motion scaling, overflow/NaN rejection without advancing history,
authenticated styles, alpha, schedule changes, CPU/GPU switching, model
replacement, caller-context restoration and foreign-adapter refusal. D3D12 debug
validation reported no errors. Windows half conversions pass exhaustive half-code
and odd-tail checks on x86/x64; the SDK chain CPU-oracle regression also passes.

Linux software contracts and final CUDA host-source compilation remain covered;
Linux GPU inference is not part of these Windows results. Live fixture acceptance
and its package hashes are recorded separately in `optimization/VALIDATION.json`.
Use `tools/run_cuda_live.py --features --require-gpu-textures` in the testing
project to require the shared-texture leaf through the installer/control addons.

### Outer capture/return GPU mailbox

The x64 addon can now keep the outer DX11/DX12 capture and return on the GPU,
using two shared NT-handle RGBA8 textures and the existing asynchronous worker.
The producer's transfer fence must retire before the worker can read its input;
the worker's private D3D12 queue completes before the render side copies its
output. The next producer epoch also retires that output copy before reuse.
Only one input epoch is admitted at a time; the CPU route retains its existing
prefetch policy. Model loading, shader compilation and inference stay on the
worker. Imported allocations survive render-side teardown until the old worker
retires, including resize and route changes.

This mailbox is independent of the selected portable neural backend. Native CUDA
can combine it with the shared-texture neural leaf; Portable CUDA and other
portable backends may still transfer pixels at their own neural boundary. The
mailbox does not make CPU inference GPU-resident. `capture_pixel_transport` and
`capture_host_transfer_bytes` report outer transport separately from each leaf's
`pixel_transport` and `neural_host_transfer_bytes`.

Admission currently requires the full x64 addon, the shared Windows pipeline SDK,
a fence/share-capable DX11 or DX12 renderer, and the enhanced temporal/services
view on an RGBA8 SDR canvas. Comparisons/remapping, BGRA/RGB10, nonidentity host
color/resolution processing, other graphics APIs and the packed spatial route
retain CPU transport. `HelperTransport=readback` explicitly forces CPU staging
for an A/B comparison. Live switches retire and recreate the appropriate bridge.
A resource-export failure falls back before inference; evaluation failures are
reported without retrying an uncertain GPU generation.

Screenshot-pair requests read back their matched input/output on demand. Regular
mailbox frames do not map pixel buffers. Screenshot readbacks are outside the
normal capture/return byte counters. Use the testing runner's
`--features --require-gpu-mailbox` to enforce this outer transport; add
`--require-gpu-textures` to require the Native CUDA leaf's GPU transport too.
Current results and limitations are recorded separately under
`runs/windows-linux-after/outer-transport` in the testing project.

DX11 NT texture allocation uses the native OS sharing interfaces, including games
that request feature level 11.0. ReShade's resource-export capability is narrower
at that feature level; this path does not request its unsupported export flag.
The same retirement fence and producer/consumer ownership rules still apply.

The fixed-sequence transport test compares the host codec/pipeline against the GPU
mailbox on the identical 1280x720 image, using a real DX11 feature-level-11.0
export and D3D12 imports. Three paired trials per backend/model each warm up five
calls and measure fifteen calls in alternating order. Timings include the worker
SDK/format conversion; game capture/presentation and the test's comparison
readbacks are excluded. Desktop GPU clocks are unmanaged.

| Backend / model | Host worker SDK | GPU mailbox worker | Ratio |
| --- | ---: | ---: | ---: |
| native-cuda / small-fp8 | 18.851 ms | 5.662 ms | 3.33x |
| native-cuda / original | 21.555 ms | 6.997 ms | 3.08x |
| cuda / small-fp8 | 87.359 ms | 73.997 ms | 1.18x |
| cuda / original | 101.945 ms | 88.681 ms | 1.15x |

All 336 compared frames match byte-for-byte, including controls and source alpha.
Requested screenshot pairs and D3D12 debug-layer checks pass. Live acceptance is
recorded separately in the testing project's `outer-transport/VALIDATION.json`;
live camera-motion throughput is distinct from these fixed-input worker times.
Other portable backends are not hardware-qualified by this CUDA validation.
The DX11 native sharing path is grounded in the [Microsoft shared-resource
contract](https://learn.microsoft.com/en-us/windows/win32/api/dxgi/nf-dxgi-idxgiresource-getsharedhandle);
[ReShade's capability check](https://github.com/crosire/reshade/blob/main/source/d3d11/d3d11_impl_device.cpp)
explains why its resource-export abstraction excludes this feature level.


## Portable CUDA, Vulkan and CPU processing

Portable CUDA and Vulkan keep their own neural providers and broker sessions.
Their D3D12 texture adapter can convert raw half-float snapshots to canonical
RGB/motion planes with compute shaders, then pack the result on the rendering
GPU. This retains host-plane neural transport; it is not the Native CUDA shared
texture executor. Raw buffer packing preserves every alpha bit and explicit
nearest-even rounding preserves half subnormals and midpoint ties.

The GPU codec is the default for CUDA/Vulkan. Set `gpu_pixel_codec: false` in a
portable-runtime configuration to select the runtime-dispatched F16C host codec
for an A/B comparison. CPU processing uses the host codec by default. Missing
advanced broker planes are skipped according to their presence flags; present
planes are fully copied before the request is published.

Automatic portable CUDA tuning on SM120 selects 64-column FP16 matrix tiles and
2048-window batches. NVIDIA Vulkan uses 2048-window batches; the tested AMD
0x13c0 integrated device uses 128. Explicit `cuda_matrix_columns` and
`window_batch_limit` overrides remain available. Optional factory/hint symbols
preserve compatibility with older providers.
CUDA also retains the seeded noise plane across matching frames, regenerating
and uploading it when its seed or graph changes. Its scratch lifetime spans the
whole graph. `device_noise_cache: false` disables this optimization for comparison.

The Windows CPU provider uses contiguous half-tile loads for AVX-512 matrices,
retains ascending reduction order, and publishes eligible residuals directly.
Its workers wake once per graph and synchronize operator dependencies through
a barrier instead of returning through a coordinator mutex after every command.
These changes retain all blocks, passes, controls and model arithmetic.

Portable frame preparation and resolve reuse bounded per-session scratch planes.
Committed and pending histories occupy separate banks and swap only after a
successful evaluation. Reset clears history validity while retaining capacity;
inactive prior planes still contain canonical zeros. Neutral/deferred display
resolve uses runtime-dispatched F16C with the scalar arithmetic and half-rounding
order preserved. Other display controls retain their full composition path, and
the recovered output profile skips discarded legacy display calculations.
This reduces host allocation and conversion overhead for CUDA, Vulkan and CPU;
it does not replace their neural kernels or remove host-plane transport.

`tools/test_portable_pixels.py --gpu --output REPORT.json` checks exhaustive
half/alpha conversion, midpoint rounding, odd pitches, motion presence/scaling,
and D3D12 debug-layer errors. `tools/test_portable_execution.py` checks CPU/CUDA
operators and local/broker advanced-plane transitions with external model
configurations. The testing project retains the common-boundary before/after
receipts under `runs/portable-optimization-current`; setup and model loading are
excluded, while upload, inference and completed readback are included.
The additional scratch/Native CUDA validation is in the testing project's
`runs/gpu-extra-mile`. `benchmark_processing_common.py --compare-runtime PATH`
interleaves a hash-verified model-free pre-change runtime with the current CUDA
and RTX Vulkan routes. Each receipt distinguishes the compiled source snapshots
and actual binaries; setup, output validation and hashing remain outside timing.
The final all-engine timing receipt is `runs/latest-gpu-extra-mile/final/run.json`.
`tools/benchmark_native_cuda_restore.py` separately measures resident GPU-texture
calls before/after workload changes; its boundary excludes host upload/readback
and must not be mixed with the common host-boundary chart.
