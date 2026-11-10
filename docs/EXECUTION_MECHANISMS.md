# Execution Mechanisms

Private DX12 contexts created inside DX11 or staged legacy routes resolve their
queue to a native identity before creating evaluator objects. ReShade's explicit
native-object COM interface is preferred. An already-native queue implemented by
the Windows system D3D12 runtime is also valid: OpenGL frontends need not hook
private DX12 creation or emit ReShade queue events. Older proxies use the exact
observed queue match. Unknown/ambiguous proxies are rejected. These checks run
only at session creation. Wrappers stay alive for the session, while the observer
and driver see one native implementation.

The private staging resource owner also registers that exact native queue for
the evaluator's submission checks. It releases the registration only after the
bridge and its GPU work retire, including constructor-failure cleanup. A GPU
timeout retains the owner and registration until process exit rather than
allowing a new route over potentially live work. This explicit lease covers
frontends that never emit ReShade queue init/destroy events; arbitrary engine
queues still require their ordinary observed lifetime.

Phoenix also resolves its shader compiler from the Windows system directory.
Older games can load an app-local `D3DCompiler_47.dll` with an older optimizer;
using it for Phoenix shaders can cause compilation failures that do not occur
in the helper. The game's compiler file remains untouched. Compilation still
uses the requested optimization level and warnings-as-errors; compiler lookup
is cached and adds no per-frame work.

Live strength, history and helper-motion edits preserve an already selected
fallback route. They do not clear a previous device/queue failure and repeatedly
cold-start the helper. Renderer changes and explicit retries can reopen rejected
routes. Model/resource changes may still rebuild the resources they affect.

For current model/backend availability and the Original-only x86 driver-host restriction, see [the support matrix](SUPPORT.md). [Current offline two-model benchmarks](benchmarks/current/README.md) measure native and portable inference separately; they are not new measurements of every transport mechanism.

In-process x86 routing checks the address space the game can actually allocate:
normally 2 GiB, or up to 4 GiB for a large-address-aware executable on supported
Windows. Free addresses above that ceiling do not count as game headroom. The
native library checks again after driver/model initialization and periodically
on its processing worker, since level loads can exhaust an initially healthy
process. If the reserve becomes insufficient, automatic routing retires that
worker and selects a compatible fallback. This preserves eligible internal
routes without assuming that loading an x64 library gives the x86 game more
usable addresses. Executables and their address-awareness flags are not edited.

The native ready signal includes the selected model's initialization and weight
allocations, not merely D3D12 device creation. The caller checks headroom after
that signal and immediately after the first frame, which can initialize motion
and format resources lazily. Subsequent checks stay on the processing worker.

Processing threads also own their stack budgets instead of inheriting the game's
executable default. Small-stack legacy hosts can reserve as little as 192 KiB,
which is insufficient for driver/model setup. The frontend worker reserves
4 MiB with on-demand page commitment; the native x64 worker reserves 8 MiB with
a 256 KiB initial commit. Their shared-state and orderly-retirement contracts
are unchanged. The small-host-stack CPU regression runs in both architectures,
and native model probes also exercise a 192 KiB host default.

The compatibility panel's **Advanced transport** section selects where processing
runs, independently of the game graphics API. It appears below the renderer
controls, so renderer selection comes first. UI choices use descriptive names,
not the investigation numbers used in the technical tables below. The x64 helper is retained.
The current development routing selects eligible same-process implementations
before the helper. The dropdown shows only mechanisms eligible for the current
architecture, canvas, settings and installed runtime. A saved unavailable choice
remains visible as a warning; it is not silently replaced.

Automatic preference order:

1. Native x64 inference in the game process: same-device D3D12; same-adapter
   DX11/DX12 shared GPU textures; or fenced CPU capture/return for DX9, DX10,
   OpenGL and Vulkan. All use the existing native evaluator and renderer controls.
2. Matching x64-library-inside-x86 package.
3. Matching x86 driver-host package (Original model, all eight blocks once,
   large-address-aware executable).
4. Matching portable runtime and an eligible configured device.
5. External x64 helper.

Automatic motion estimation is enabled by default. Valid engine motion vectors
take priority; missing vectors are estimated at the configured analysis size.
Eligibility includes the requested frame services, not only the renderer. A
matching x64-library-inside-x86 package can estimate motion and maintain temporal
history in the game process, for both Original and Lightweight FP8. Automatic
can select it with motion enabled; it no longer needs a helper just for motion.
The package must declare the matching frame-service contract and pass its
existing Windows/driver, model and address-space checks. Older packages, the
x86 driver-host route and the legacy x86 portable SDK remain spatial-only;
Automatic uses the helper when no eligible route supplies the requested services.
The x64 native routes also provide motion services without a helper.

A matching frame-service contract proves that the runtime implements the
requested services; it does not establish whole-game stability. Keep the x86
library route experimental. A small standalone probe does not reproduce a
game's address-space pressure or level-loading behavior. Caught allocation
failures can fall back to the helper, but a process-level fault inside the game
cannot be recovered by that fallback.

The x86 library route carries capture identity, timestamps, reset controls and
live motion settings into the shared native SDK. Its CPU capture does not supply
engine vectors: Automatic estimates missing vectors rather than treating a zero
texture as valid engine data. The overlay reports the actual provider, analysis
size, valid-vector frames and history resets. Analysis-size/provider edits reset
the affected history without replacing the worker. Cold initialization can also
cross the existing one-second discontinuity threshold and require another
priming frame; subsequent uninterrupted frames retain their history.

This temporal x86 route currently requires one evaluation per captured frame.
Custom chains, adaptive priority, unsupported appearance controls and temporal
refinements retain the helper fallback. A spatial-only selection can still use
older eligible routes. No weights are added to the bridge package.

## Native x64 capture routes

| Game API | Native inference | Transfer | Diagnostic execution mode |
| --- | --- | --- | --- |
| DX12 | Game process, game device and DIRECT queue | Same-device GPU resources | `in_process` |
| DX11 | Game process, DX12 device on the game adapter | Shared NT texture and fences, no CPU pixels | `in_process_d3d11` |
| DX9/10, OpenGL, Vulkan | Native DX12 worker in the game process | Existing fenced CPU capture/upload and return | `in_process_staged` |

The DX11 surface owns the shared texture, both API views, a DIRECT queue and a
shared fence. Game input is copied before inference; the output fence orders
writeback before Present. Predication is restored after unconditional copies.
The evaluator is the same `D3D12Bridge` used by DX12 games, including model
selection, NVIDIA DLSS5/comparison, live controls and completed-frame holds.

The staged route retains `ReadbackBridge` for API-specific capture/presentation
and Vulkan submission ordering. Its worker owns upload/readback buffers and the
native evaluator; no external process or window is created. Initialization-only
responses are retried on the same captured image, with a bounded deadline, and
never published as neural output. `frame_wall_ms` includes upload, inference and
readback; same-process execution does not make those transfers free.

Only one native feeder generation owns the shared session at a time. Stop
requests retain worker/module/resource ownership until retirement. GPU timeouts
pin uncertain allocations and stop processing instead of falling back over live
work. Unsupported adapters, layouts or settings can fall back after a safe
failure; an explicit internal choice stays strict. Protected input is refused.

Regression coverage includes WARP DX11/DX12 texture/fence round trips, RGB8
staging, synthetic renderer switching, initialization/deferral holds, local
worker PID/output/tuning/shutdown checks and API/architecture selection. These
are software and CPU checks, not new model-quality or retail-game measurements.

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

Low 32-bit address space and host `bad_alloc` failures select the external x64
helper directly instead of trying another renderer inside the same exhausted
process. This applies to Automatic and fallback-enabled selections; strict
selections retain their failure. The original error and recovery decision are
written to `ReShade.log` and retained in live status. GPU/VRAM exhaustion does
not trigger this policy, and helper startup/inference failures remain errors.

The 2026-10-20 development build passed presented-output checks in generated
640×360 DX12 fixtures for x86 library, x86 driver, portable fallback from a
configured external folder, final helper fallback and x64 same-device execution.
The checks require completed inference, a fresh presented revision and a measured
pixel change. They do not establish retail-game compatibility or output parity.
Worker wall-time medians were 16.96 ms (library), 18.14 ms (driver), 31.54 ms
(portable) and 5.23 ms (helper); the x64 same-device route lacks that counter.
These diagnostic samples show why "same-process first" does not mean "fastest".

For an x86 game, **External x64 helper** with automatic/shared DX11 transfer can
be considerably faster than a native route that captures, converts and composes
pixels on the CPU. GPU inference time alone excludes that CPU work; compare the
worker's full frame wall time too. Native CPU transports bypass identity sRGB
processing at matching extents and unit exposure; actual scaling, HDR and
exposure changes still use the complete conversion pipeline. Renderer strengths
and comparison remain active when that identity conversion is skipped.

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
| In-process native GPU | Optimized native executor inside the x64 game | DX12 same-device resources; DX11 shared GPU textures; DX9/10/OpenGL/Vulkan CPU capture/return | Requires supported device/format/executor; refuses unsupported combinations |
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
  binds the private native TLS routines to the verified Windows image. The
  installed NVIDIA CUDA driver is discovered and validated separately; a
  build-machine driver directory/hash is not an eligibility requirement.
  A Windows/package mismatch or failed driver compatibility check is refused
  before inference, with the actual failure retained in diagnostics.
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
      bootstrap.json             # verified Windows bootstrap + package hashes
      phoenix_driver_probe32.exe # disposable model-free driver compatibility check
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
working motion provider, evaluation resets history. The x64 shared frame worker,
including the qualified x64 library inside an x86 game process, can estimate
motion from consecutive source images and reject low-confidence history; it does
not manufacture geometric depth.

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
live tone/structure/skin/mask conditioning. Legacy spatial packages use reset-frame
inference; frame-service-capable route 2 can preserve motion-aligned history. Iterations
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

### Installed-driver discovery and validation cache

The experimental x86 native bridges discover the registered `nvlddmkm` package
under the Windows driver store and select its x64 `nvcuda64.dll`. They do not scan
old packages and guess which one is newest. x86 readers use the native system
view; the x64 worker receives the corresponding native loader path. No driver
files, registry values or expected hashes are rewritten.

On a cache miss, a disposable x86 process starts the native worker with the same
Windows TLS and UMD adaptations as the bridge. It checks CUDA initialization and
device enumeration, matches the CUDA device to an NVIDIA DXGI adapter, resolves
the required NVAPI interfaces, and verifies a tiny D3D12 queue/fence/copy/readback.
It loads no model and performs no neural inference. Its job closes on failure or
timeout, with a 20-second limit; it is not a persistent inference helper.

Results live in `%LOCALAPPDATA%/PhoenixDLSS5/DriverCompatibility`. The cache
identity includes the driver path/content hash, native Windows image hash,
bootstrap/package identity (including the probe and bridge hashes), and probe
contract. Driver or package changes require fresh validation. Successful checks
are reused; failed checks expire after two minutes so transient initialization
errors can recover. Corrupt, incomplete or mismatched records are not accepted.
Expiry is evaluated on the next availability/startup check or **Recheck runtimes /
retry routes** action; the cache does not force periodic renderer restarts.
Native startup rechecks the selected driver, independently of route discovery.
Validation and hashing occur during background availability/startup, not per frame.

This qualifies driver startup and the mixed-bitness graphics bridge, not every
game, model, kernel style or future Windows build. Actual renderer initialization
and normal failure/fallback handling remain required. Windows hash/timestamp and
private-offset checks remain in place. `native_driver_validation` in the runtime
status reports the qualified CUDA hash and whether the result came from cache.

`python tools/test_wow64_driver_discovery.py` runs x86/x64 CPU cache, path and
process-cleanup tests. `python tools/validate_dynamic_driver.py --runtime PATH
--allow-gpu-startup` explicitly runs the isolated startup/copy probe and verifies
cache reuse. Neither command launches a game; the latter uses the GPU without
performing model inference. Rebuild the optional bridge package together with
the add-ons; older packages without the dynamic probe report that an update is
needed and retain helper fallback.

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

Comparison layout, divider position and difference gain are per-frame live controls on the helper as well as resident backends. The game UI and control mailbox use the same resource-change policy; changing these controls keeps the loaded helper/model. Renderer/source selections that require an additional NVIDIA feature remain resource changes. Tests validate the live-patch whitelist and preserve rebuilds for model/feature changes. Runtime continuity receipts distinguish a stable process from a restarted helper; package tests alone do not establish this.

### CPU transport work in legacy graphics routes

An x64 neural library inside an x86 game still has a CPU pixel boundary when
its graphics API cannot share GPU resources. The legacy route captures RGBA8,
packs RGBA16F, invokes the x64 worker, decodes
its half-float output to RGB and composes RGBA8 for the game's upload. That work
must not be confused with GPU inference time or the game's presentation FPS.

The CPU path uses immutable256entry byte-to-half and65,536entry half-to-float
tables generated by the existing DirectX conversion functions. A second64KiB
table fuses half-to-UNORM8 conversion for the plain enhanced-only view. Together
they add about321KiB per process, preserve finite half values bit-for-bit, and reject every
infinite/NaN RGB encoding. Alpha remains governed by the original input.
The ordinary enhanced-only view bypasses general comparison/remapping work;
zoom, wipes and difference modes retain the general composition path. UNORM8
rounding retains the previous float-product/lround contract, including adjacent
values around half-integer thresholds. An unused full-frame RGB conversion is
also omitted when native inference supplies that RGB result itself.
For the one-pass enhanced-only view at unchanged geometry, the WOW64 frontend
returns RGBA8 directly from the half output, eliminating a second full RGB32F
allocation. Comparisons, magnification, resizing and repeated network passes
retain the RGB32F path. Input packing uses an AVX2/F16C conversion on CPUs whose
CPUID and OS extended-state support permit it; its isolated dispatched function
does not raise the baseline CPU requirement. Older CPUs keep the lookup path.
Exhaustive byte/half and rounding tests cover both representations.
The dispatched return conversion handles two RGBA half pixels at a time,
validates only RGB, retains the input alpha, and explicitly reproduces the
existing half-up UNORM8 rule instead of using a different SIMD rounding mode.

Staging capture keeps one frame ahead of the worker with one bounded CPU queue.
DX9 additionally requires an optional second GPU
snapshot can be allocated. The snapshot for the in-flight model input remains
separate from the next capture; it is promoted only after the previous completed
input has been copied into the composed pair. The CPU queue is bounded to one
retired image. Screenshot inputs are remembered at submission, not at prefetch,
so the next capture cannot replace the current output's paired input. If the
extra texture allocation fails, DX9 capture remains serialized as before. Other
staging APIs do not need that second texture: their current input already lives
in independent CPU storage while the next GPU capture retires. Transfer
telemetry exposes `capture_prefetch` and`queued_input`; worker duration and fresh
output FPS still describe different parts of the pipeline.

CPU-only numerical/performance probes motivated these changes. They do not
establish in-game FPS, GPU speedup, or that a previously installed package already
contains the new source. Validate a rebuilt package before qualifying a game.
`cpu_color_transfer_tests` runs in the normal add-on build and the focused
`tools/test_model_selection.py` CPU checks for both x86 and x64.

### Compact pixels for the same-process x64 worker

A matching worker can advertise `library_rgba8_transport=1` in its hash-bound
`bootstrap.json`. For a single-pass, unchanged-size Phoenix DLSS5 enhanced view,
the frontend then sends RGBA8 directly. The native worker uploads those bytes,
converts to the model's half-float textures on the GPU, evaluates with the same
motion/history settings, and quantizes back to bytes on the GPU. This avoids
CPU half conversion and halves the upload/readback payload compared with RGBA16F.
The upload and evaluation use the same D3D12 queue; queue ordering establishes
their dependency without an extra CPU upload wait. Allocator and resource reuse
still wait for completion. This remains in the game's process.

The GPU conversion explicitly preserves the prior nearest-even input half
rounding, half-up SDR byte output and input alpha. An additional GPU status word
rejects nonfinite RGB before any output is published. The renderer returns one
CPU output buffer after its screenshot/upload copies are finished; the native
worker reuses that buffer instead of allocating and clearing a full image for
every frame. This return slot is bounded to one image and protected by the worker
mutex. It never lends a vector still referenced by a GPU upload or screenshot.

Older profiles that do not advertise this capability retain their existing
CPU-half route. Comparison/refinement float outputs, NVIDIA DLSS5 selection and
the native x86 driver route also retain that path. The packet size is unchanged;
only a negotiated former reserved word selects compact pixels. Buffer ranges,
extents, overlap and transport/backend combinations are checked by the worker.
Per-frame pointers reset when switching back to float output.

Telemetry identifies `pixel_transport` as `rgba8_gpu` or `rgba16f_cpu` and reports
`output_buffer_reused`. Detailed monitoring additionally reports CPU wall stages
under `transfer_stages_ms`: upload/enqueue, evaluation/wait, and pack/readback.
These are different from the SDK's narrower GPU interval. An upload/enqueue
sample does not imply that the GPU upload has completed at that timestamp.

`python tools/test_wow64_pixel_transport.py` runs the x86/x64 CPU contract checks.
The explicit `--gpu` option adds isolated D3D12 checks for exact input half bits,
finite half output quantization, source alpha, odd pitches, invalid-output
rejection and error-word recovery. No game or neural model is loaded by this
test. `wow64_motion_tests.cpp` additionally exercises alternating packed/float
frames, live motion settings, history reset, buffer reuse and clean retirement
with an explicitly supplied local model and runtime.

The bounded DX9 capture FIFO and optional same-submission native output path are
documented in [Frame delivery and worker overhead](FRAME_DELIVERY_OPTIMIZATION.md).
Its validation distinguishes isolated scheduling/pixel checks from actual-game
delivery rates; a fast worker alone is not proof of fast displayed output.

CPU pixel transport has no game-adapter affinity. The external native helper selects a hardware NVIDIA adapter in high-performance order, rather than the Windows default adapter, which may be an AMD or Intel integrated GPU. It logs the selected adapter. A shared-texture helper keeps the explicitly requested adapter identity. If no NVIDIA device is eligible, native helper startup fails with a specific adapter message. NVAPI failures include their status code and call location.
