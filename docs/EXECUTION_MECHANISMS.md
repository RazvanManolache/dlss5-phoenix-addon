# Execution Mechanisms

For current model/backend availability and the Original-only x86 driver-host restriction, see [the support matrix](SUPPORT.md). [Current offline two-model benchmarks](benchmarks/current/README.md) measure native and portable inference separately; they are not new measurements of every transport mechanism.

The compatibility panel's **Advanced transport** section selects where processing
runs, independently of the game graphics API. It appears below the renderer
controls, so renderer selection comes first. UI choices use descriptive names,
not the investigation numbers used in the technical tables below. The x64 helper is retained.
Installing an experimental portable runtime does not change automatic routing.

**0.9 downloads:** the standard add-on and optional portable runtime packages
include no profile-specific WOW64 bundle. The native x86 experiments below retain
their earlier evidence, not fresh 0.9 Windows/driver-profile certification. They
require a separately built and verified matching runtime. Automatic/helper
transport remains available without it.

## Requested Native Routes

These numbers refer to the requested experiments, not the menu order above.

| Number | Mechanism | Status |
| --- | --- | --- |
| 1 | x64 Phoenix DLSS5/NVIDIA DLSS5 inside an x64 application | Working; native DX12 fixture benchmarks retained separately |
| 2 | x86 application loads the x64 Phoenix DLSS5/NVIDIA DLSS5 library in the same process | Integrated as an opt-in x86 add-on route; packaged client inference, live strengths, repeated passes and backend switching verified; game presentation not yet validated |
| 3 | x86 Phoenix DLSS5 host calls the native driver through a same-process x64 driver thunk | Integrated as an opt-in x86 add-on route; packaged client inference matches route 2 exactly; Phoenix DLSS5 only, full eight-block network; game presentation not yet validated |
| 4 | x86 application communicates with a separate x64 helper | Working and retained |

The portable x86 Vulkan/SPIR-V implementation is an **additional fallback, not
route 3**. Its benchmark does not measure the requested native driver bridge.
The earlier driver-only route-3 prerequisite probe creates modules only. A
separate full x86-host probe now authenticates the model, generates the native
graph and its dispatch arguments in x86, then evaluates it through a generic
x64 driver executor in the same process. It loads no x64 Phoenix DLSS5 renderer and
uses no helper or Vulkan. Reset-frame inference and live strength changes pass;
game presentation, temporal integration and automatic selection remain
unvalidated. See [route-3 host results](WOW64_EXPERIMENT.md#route-3-x86-host-inference).
See [native driver investigation](X86_NATIVE_INVESTIGATION.md#direct-driver-bridge).

| Mechanism | Processing | Capture / return | Selection policy |
| --- | --- | --- | --- |
| Automatic | Existing native D3D12 route when eligible; otherwise x64 helper | Same-device D3D12, D3D11 shared textures, or staging | Conservative existing default |
| In-process native D3D12 | Optimized native executor inside the game | Same-device GPU resources | Requires supported D3D12/device/format/executor; refuses unsupported combinations |
| External x64 helper | Existing graphics companion | Automatic, D3D11 shared textures only, or staging/CPU pixels | Always available as a selectable alternative |
| In-process portable (experimental) | Matching-bitness C++ runtime on a worker thread inside the game | Existing fenced readback and upload | Opt-in; does not use the failing x86 native NVAPI path |
| x64 library inside x86 (experimental) | Complete native x64 Phoenix DLSS5/NVIDIA DLSS5 renderer on a native thread in the x86 game | Existing fenced CPU capture and return | Opt-in x86 only; locally matched Windows/driver bootstrap package required |
| x86 Phoenix DLSS5 + native driver (experimental) | x86 graph/model/controls host with generic x64 driver executor in the same process | Existing fenced CPU capture and return | Opt-in x86 only; Phoenix DLSS5 full network; requires a large-address-aware executable; never patches the game EXE |

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
Returning to a Phoenix DLSS5-only view restores the selected route. Route 2 can
evaluate both renderers in the same x86 process. Automatic/native x86 validation
guards are unchanged; selecting an experimental route never makes it automatic.

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
remain available. Backbuffer input has no engine motion/depth, so every evaluation
resets temporal history rather than pretending guides exist.

This first portable backbuffer adapter uses normal-resolution SDR host planes.
Super-resolution and adaptive blocks are disabled for it; incompatible saved
settings fail explicitly and can be reset in the panel. Native NVIDIA DLSS5 host
exposure/jitter/HDR declarations, override-file reload and offline input-capture
bursts are not portable-runtime features. Unsupported runtime actions are
rejected, not falsely acknowledged. Standard game screenshots remain separate.

Status exports distinguish `in_process`, `in_process_portable`,
`in_process_x64_library`, `in_process_x86_driver`, `helper`, and
`not_started`. All in-process execution reports the game's PID as `inference_pid` and
zero `helper_pid`. Its wall time includes host preparation/inference/resolve and
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
Changing to a different NVIDIA DLSS5 model DLL after one has been loaded is explicitly
refused by route 2, rather than continuing with the old library under a new label.
With fallback enabled the helper handles that selection; otherwise restart the
game. Phoenix DLSS5-only weight/model selection remains independent of this DLL limit.

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

The separate [x86 Vulkan validation](X86_VULKAN.md) covers real standalone neural
execution. Both layers of evidence are needed before promoting this mechanism
to automatic selection.
