# Execution Mechanisms

For current model/backend availability and the Original-only x86 driver-host restriction, see [the support matrix](SUPPORT.md). [Current offline two-model benchmarks](benchmarks/current/README.md) measure native and portable inference separately; they are not new measurements of every transport mechanism.

The compatibility panel's **Advanced transport** section selects where processing
runs, independently of the game graphics API. It appears below the renderer
controls, so renderer selection comes first. UI choices use descriptive names,
not the investigation numbers used in the technical tables below. The x64 helper is retained.
The current development routing selects eligible same-process implementations
before the helper. The dropdown shows only mechanisms eligible for the current
architecture, canvas, settings and installed runtime. A saved unavailable choice
remains visible as a warning; it is not silently replaced.

Automatic preference order:

1. Same-device native D3D12 in x64.
2. Matching x64-library-inside-x86 package.
3. Matching x86 driver-host package (Original model, all eight blocks once,
   large-address-aware executable).
4. Matching portable runtime and an eligible configured device.
5. External x64 helper.

Runtime package and Windows/driver hash checks run off the rendering thread.
The driver host additionally needs the matching tile-aligned graph and its pinned
digest. Older native packages without a model-support declaration are eligible
for Original only; selecting another model needs a package declaring support.
Automatic waits for discovery, then skips absent/incompatible routes. A failed
route is excluded for that setup and the next eligible mechanism is attempted;
the UI exposes failure history and a recheck/retry action. Unretired GPU work
still requires a restart rather than an unsafe fallback. This order is a
preference for avoiding external transport, not a universal speed claim.

Explicit library, driver and portable selections also use this recovery order
when their native/portable fallback checkbox is enabled. This includes startup
availability refusals such as insufficient 32-bit address space, not just errors
after executor creation. Disabling fallback keeps the explicit route strict.
Discovery and graphics resource/queue safety checks still apply.

The 2026-10-20 development build passed presented-output checks in generated
640×360 DX12 fixtures for x86 library, x86 driver, portable fallback from a
configured external folder, final helper fallback and x64 same-device execution.
The checks require completed inference, a fresh presented revision and a measured
pixel change. They do not establish retail-game compatibility or output parity.
Worker wall-time medians were 16.96 ms (library), 18.14 ms (driver), 31.54 ms
(portable) and 5.23 ms (helper); the x64 same-device route lacks that counter.
These diagnostic samples show why "same-process first" does not mean "fastest".

**Public downloads:** the standard add-on and optional portable runtime packages
include no profile-specific WOW64 bundle. The native x86 experiments below retain
their earlier evidence, not fresh Windows/driver-profile certification. They
require a separately built and verified matching runtime. Automatic/helper
transport remains available without it.

Current native x86 packages use `phoenix-wow64-bootstrap/3` for coordinated TLS
and worker restart handling. Older `/2` packages are deliberately refused: rebuild
the profile-bound runtime with the matching current source rather than mixing
old bootstrap DLLs with the new add-on. The x86 preflight checks both total free
address space and the contiguous region needed by the selected route; its safety
margins are startup guards, not steady-state RAM estimates. Low-space routes
remain retryable, and the x64 helper is still available.

## Requested Native Routes

These numbers refer to the requested experiments, not the menu order above.

| Number | Mechanism | Status |
| --- | --- | --- |
| 1 | x64 Phoenix DLSS5/NVIDIA DLSS5 inside an x64 application | Working; native DX12 fixture benchmarks retained separately |
| 2 | x86 application loads the x64 Phoenix DLSS5 library in the same process | Phoenix-only matching-package development route; NVIDIA DLL evaluation requires the helper; wider game compatibility remains unqualified |
| 3 | x86 Phoenix DLSS5 host calls the native driver through a same-process x64 driver thunk | Matching-package development route; presented Phoenix output verified in a generated x86 DX12 fixture; Original model and full eight-block network; wider game compatibility remains unqualified |
| 4 | x86 application communicates with a separate x64 helper | Working and retained |

The portable x86 Vulkan/SPIR-V implementation is an **additional fallback, not
route 3**. Its benchmark does not measure the requested native driver bridge.
The earlier driver-only route-3 prerequisite probe creates modules only. A
separate full x86-host probe now authenticates the model, generates the native
graph and its dispatch arguments in x86, then evaluates it through a generic
x64 driver executor in the same process. It loads no x64 Phoenix DLSS5 renderer and
uses no helper or Vulkan. Reset-frame inference and live strength changes pass.
The newer fixture checks above add presentation evidence; temporal correctness
and wider gameplay remain unqualified. See [route-3 host results](WOW64_EXPERIMENT.md#route-3-x86-host-inference).
See [native driver investigation](X86_NATIVE_INVESTIGATION.md#direct-driver-bridge).

| Mechanism | Processing | Capture / return | Selection policy |
| --- | --- | --- | --- |
| Automatic | Eligible same-process routes in the order above, then x64 helper | Same-device D3D12, D3D11 shared textures, or staging | Same-process preference; checks package readiness and settings |
| In-process native D3D12 | Optimized native executor inside the game | Same-device GPU resources | Requires supported D3D12/device/format/executor; refuses unsupported combinations |
| External x64 helper | Existing graphics companion | Automatic, D3D11 shared textures only, or staging/CPU pixels | Always available as a selectable alternative |
| In-process portable (experimental) | Matching-bitness C++ runtime on a worker thread inside the game | Existing fenced readback and upload | Offered when provider/device requirements pass |
| x64 library inside x86 (experimental) | Native x64 Phoenix DLSS5 renderer on a native thread in the x86 game | Existing fenced CPU capture and return | x86 only; Phoenix-only; locally matched Windows/driver bootstrap package required |
| x86 Phoenix DLSS5 + native driver (experimental) | x86 graph/model/controls host with generic x64 driver executor in the same process | Existing fenced CPU capture and return | x86 only; Phoenix DLSS5 full network; requires a large-address-aware executable; never patches the game EXE |

The two experimental native x86 routes are selectable in **Advanced transport**.
They use a constrained high-address reservation retry and early native CUDA
initialization, in addition to the native TLS/UMD/GDI bootstrap. Native worker
recreation allocates loader-managed TLS for already-loaded native libraries.
See the
[latest WOW64 investigation](WOW64_EXPERIMENT.md#latest-result-native-inference-works-in-the-probe).
This does not establish loader/device teardown safety in arbitrary games or
change the helper fallback. The independent x64 portable Vulkan probe can create an instance
but sees no GPU even through direct ICD/adapter-LUID enumeration. These
diagnostics do not add a fifth working route or change automatic selection.

Portable capture is wired for the existing D3D9, D3D10, D3D11, D3D12, OpenGL and
Vulkan readback paths, in x86 and x64 builds. This is source integration, **not
new game-presentation validation**. Supported input is a single SDR RGBA8/BGRA8
canvas (also SDR D3D12 RGB10A2), up to 8192 pixels per axis. D3D9 capture can resolve
MSAA; other portable transports require single sampling. CPU pixel copies remain:
in-process does **not** mean zero-copy or guaranteed better performance.

For portable and route 3, NVIDIA DLSS5 DLL evaluation and comparisons containing
NVIDIA DLSS5 use the retained x64 helper while the saved preference remains intact.
Returning to a Phoenix DLSS5-only view restores an eligible selected route.
Route 2 is also Phoenix-only: repeated x86 NVIDIA trials exposed model-image
mapping failures, host allocation failures and access violations. It is excluded
from NVIDIA/comparison transport choices. Automatic selects the x64 helper;
an explicitly saved route-2 NVIDIA selection is refused with an explanation,
without silently changing that saved preference. The ordinary same-device native
D3D12 route remains x64-only. Automatic can select the separate x86 routes when
their installed package, Windows/driver profile and settings are eligible.

## Configuration

Settings are saved under `[Phoenix.Compatibility]` in ReShade's configuration:

```ini
[Phoenix.Compatibility]
ExecutionMode=portable
PortableRuntime=PhoenixRuntime\win-x86
PortableDevice=auto
PortableAllowUnvalidated=0
PortableFallback=1
NativeRuntime=PhoenixRuntime\wow64
NativeFallback=1
HelperTransport=auto
CoherentFrames=1
```

- `ExecutionMode`: `auto`, `in_process`, `helper`, `portable`,
  `wow64_library` (route 2), or `wow64_driver` (route 3).
  Automatic tries eligible same-process routes first, with the helper last.
  `NativeFallback`/`PortableFallback` affect explicitly selected modes; they do
  not disable Automatic's ordered retry behavior.
- `HelperTransport`: `auto`, `shared`, or `readback`. Forced shared transfer
  requires D3D11 shared textures/fences; unsupported combinations fail visibly.
- `PortableRuntime`: absolute path, or relative to the **add-on directory**, never
  the game's current working directory. Empty defaults to
  `PhoenixRuntime\win-x86` or `PhoenixRuntime\win-x64`. An explicit
  `portable_cpp.runtime_root` in `Phoenix.json` also supplies the initial value.
- `PortableDevice`: `auto` discovers a capable Vulkan GPU on the background
  worker; `cpu` selects the reference CPU backend; explicit Vulkan/CUDA/HIP/SYCL
  GPU device IDs use the matching backend DLL. Availability depends on the
  supplied architecture-specific runtime. SYCL CPU retains its helper/broker
  requirement and is refused by the in-process portable route.
- `PortableAllowUnvalidated`: explicit experimental opt-in for a device absent
  from the runtime's `validated_devices` list. Default off. Validation catalogues
  and backend tuning options can be supplied in `phoenix_native_cpp.json` in the
  selected runtime folder; the add-on does not invent validation entries.
- `PortableFallback`: if portable startup/evaluation fails, preserve the reason
  and use the existing helper. Default on. Turning it off leaves the failure
  visible instead of silently changing mechanism. Changing a mechanism or retrying
  permits a fresh attempt.
- `NativeRuntime`: absolute or add-on-relative path for routes 2/3. Empty uses
  `PhoenixRuntime\wow64`. Contains no model DLLs or weights. `bootstrap.json`
  binds the native TLS routines to the verified Windows image and driver.
  A Windows/driver/package mismatch is refused before inference.
- `NativeFallback`: preserve a native route failure and use the retained helper.
  Default on. Off leaves processing failed visibly; the preference is retained.

All these settings have controls in Advanced transport. Device enumeration,
DLL loading, model loading and inference run off the presentation thread. A
cancelled worker owns its resources through completion; shutdown does not join it
on the render thread. A replacement portable session waits on its own worker for
the prior provider/session to retire, with cancellation and a 30-second timeout.
Concurrent contexts cannot initialize the single-session provider over each other.

## Runtime Layout

```text
game-directory/
  Phoenix.addon32                 # or Phoenix.addon64
  Phoenix.json                   # existing external model selection
  ReShade.ini
  PhoenixRuntime/
    win-x86/                     # use matching bitness; win-x64 for x64 games
      phoenix_runtime_cpp.dll
      phoenix_vulkan_cpp.dll
      phoenix_cpu_cpp.dll        # optional CPU alternative
      phoenix_native_cpp.json    # optional validation/backend configuration
      vulkan/
        phoenix_vulkan.dll
        *.spv
    wow64/                       # optional experimental routes 2/3, x86 only
      bootstrap.json             # locally verified Windows/driver profile
      wow64ext.dll                # x86, unmodified dynamic LGPL dependency
      phoenix_bootstrap64.dll
      phoenix_driver64.dll       # route 3 generic driver executor
      phoenix-nvngx.dll          # our route 2 adapter, NOT NVIDIA model code
      phoenix_native.dll         # our x64 Phoenix renderer for route 2
      native/                    # model-free authenticated kernel bundle
      licenses/
```

Use the model source already selected in the Phoenix DLSS5 panel. The runtime package
contains no weights/NVIDIA DLSS5 DLLs. Microsoft C/C++ runtime dependencies must match
the runtime build. Missing/wrong-bitness DLLs produce a diagnostic and follow the
configured fallback policy.

## Controls and Diagnostics

The portable adapter applies live tone, structure, skin/mask conditioning,
network passes and block repetitions. Comparison uses the input snapshot from
that exact evaluation, with wipe, absolute, signed, heatmap and luminance views.
Original bypass, the compatibility screenshot path, markers and timing clearing
remain available. Backbuffer input has no engine motion/depth. Without an enabled,
working motion provider, evaluation resets history. The x64 shared frame worker
can estimate motion from consecutive source images and reject low-confidence
history; it does not manufacture geometric depth.

The portable backbuffer adapter uses SDR host planes. The x64 shared SDK worker
supports independent processing/output geometry, NIS/tiny scaling, scene resets
and pipeline latency policy. GPU-time block budgets use actual backend GPU
events, independently of optional monitoring. The older equal-extent C API path
keeps its explicit restrictions. Native NVIDIA DLSS5 host
exposure/jitter/HDR declarations, override-file reload and offline input-capture
bursts are not portable-runtime features. Unsupported runtime actions are
rejected, not falsely acknowledged. Standard game screenshots remain separate.

Status exports distinguish `in_process`, `in_process_portable`,
`in_process_x64_library`, `in_process_x86_driver`, `helper`, and
`not_started`. `processing_host_pid` identifies the capture/stage host. The
shared portable pipeline reports its actual native neural worker processes in
`inference_pids`, with the first worker in the compatible `inference_pid` field
(zero when no neural stage is active). Ordinary in-process inference reports the
game's PID. Capture inside the game still has zero `helper_pid`; native neural
workers are separate from the legacy graphics helper. Wall time includes host preparation/inference/resolve and
is not labeled as GPU-only timing. Game presents and completed model outputs
remain separate counters.

Routes 2/3 share normal-resolution SDR capture/return, same-input comparison,
live tone/structure/skin/mask conditioning and reset-frame inference. Iterations
keep FP16 model input between passes rather than quantizing intermediate output
to 8-bit. Route 2 supports native block repetitions; route 3 currently requires
all eight blocks once. Unsupported resolution/adaptive/guide controls are
disabled or explicitly refused. Route-3 NVIDIA DLSS5 views use the helper regardless
of `NativeFallback`, since that route has no x86 NVIDIA DLSS5 model host.
Both native x86 routes require the Native RTX executor. CPU/portable executors
use their own route or the helper, not a mislabeled native evaluation.

These mechanisms still copy pixels through CPU staging and create their own
D3D12 device inside the game process. They are not same-device zero-copy, are not
crash-isolated, and do not inherit the resident-input benchmark's latency.
Native code stays loaded through process exit. A retiring processing worker owns
its packets and services driver callbacks until the native thread exits; the
render thread does not join it. A stuck driver may strand a processing worker,
but there is no timeout that terminates the game.
Phoenix reads the selected model's resources as data; route 2 does not execute
the NVIDIA DLL. NVIDIA DLL evaluation and model changes use the helper in x86
games. The x64 addon's same-device NVIDIA route is unaffected.

Export-relative callback stubs stay immutable once published as executable,
read-only memory. Worker retirement restores export slots, while subsequent
workers reuse the same owned stubs. `tools/test_wow64_export_patch.py` checks
this restart sequence without a GPU or system-image modification; the check
also runs in the x64 part of `tools/test_model_selection.py`.
The stub allocator searches the export's legal 32-bit RVA range rather than
assuming native images and free address space lie below 2 GiB. CPU regressions
cover a high image, a reserved 128 MiB neighborhood, and adaptation/restoration
of 300 imports (beyond the former 256-slot ceiling).
The experimental mixed-bitness loader omits optional NVIDIA overlay capture and
diagnostic clients. Diagnostic thread callbacks faulted during repeated WARP
worker restarts; the normal x64 addon/helper loader is unaffected by this rule.
It also avoids explicitly loading an absent overlay during bridge setup.

The retained experimental NVIDIA library implementation applies import adaptation to the directly loaded
NGX core and selected model DLL before initialization. Worker retirement releases
the feature and parameters, then calls device-specific NGX shutdown before
destroying its D3D12 device. A failed feature creation also retires an initialized
NGX context; model DLLs remain loaded to preserve cached driver callbacks.
This experimental implementation is not an available x86 addon route.

## Validation Boundary

`tools/validate_portable_bridge.py` runs x86/x64 CPU-only route, live-tuning,
composition, lifecycle, cancellation and failure contracts against a fake C API
runtime. `--build-addons --ngx-sdk PATH` also compiles both full add-ons, retaining
the current embedded helper payload unchanged, into a separate local candidate.
No installed games or dashboard results are modified. This does not establish
rendered game correctness, resize/device-loss safety or comparative performance.

`tools/build_wow64_runtime.py --wow64ext PATH --ngx-sdk PATH` builds the optional
native package without running the GPU or bundling model/driver DLLs. It requires
the current local bootstrap evidence to match Windows. This is deliberately not
a portable, publicly validated Windows-version catalogue.
`tools/validate_wow64_addon_client.py --run-gpu` explicitly runs the packaged x86
client and add-on worker tests with the locally selected external model. Its
receipt is `artifacts/current/wow64-addon-client-validation.json`; it is separate
from real ReShade/game-presentation evidence. No retail game is launched.
An optional `--x86-proxy PATH` copies an existing x86 ReShade DXGI proxy beside
the disposable client for the dependency-search regression. The native worker
loads dependencies from its own directory/System32, not the x86 application's
proxy DLL. This packaged-client test passes with the local x86 ReShade proxy;
it still does not establish actual swapchain presentation.
`--runtime PATH --weight-source PATH` selects an explicit local profile and model
configuration without copying the model. `--opengl` keeps a real NVIDIA x86
OpenGL context rendering and checking pixels while the probe exercises worker
creation, live controls, backend changes and retirement. This is a driver
coexistence test; the addon's x86 OpenGL native-route exclusion remains in place
until the full presentation path is qualified.

The separate [x86 Vulkan validation](X86_VULKAN.md) covers real standalone neural
execution. Both layers of evidence are needed before promoting this mechanism
to automatic selection.
