# Graphics API and 32-bit game compatibility

The full D3D12 addon can process either engine DLSS input or standalone final-color
input inside the game. Standalone processing chooses a same-device GPU route
when its runtime capabilities permit it, rather than equating missing game DLSS
calls with a need for a helper. Other APIs currently use the compatibility bridge
to a private 64-bit D3D12 helper. Full x86 builds now include the Phoenix
evaluator instead of only the transport frontend. However, local x86 native
execution is currently gated off: live D3D11-on-12 validation faulted inside the
32-bit NVAPI/graphics driver despite successful compilation and entry-point
inspection. Automatic x86 processing therefore uses the safe x64 helper. This
gate must not be removed based on compilation or function availability alone.
Both paths call the same direct Phoenix evaluator without synthetic SR/DLAA calls.

| Game | Addon file | Frame transport |
| --- | --- | --- |
| 64-bit D3D12 with engine DLSS input | `Phoenix.addon64` | Existing in-process engine route |
| 64-bit D3D12 standalone input | `Phoenix.addon64` | Same-device GPU conversion, direct Phoenix and same-frame presentation; capability-based helper fallback |
| 64-bit D3D11 | `Phoenix.addon64` | Shared GPU textures and fences |
| 32-bit D3D11 | `Phoenix.addon32` | Shared GPU textures and fences; inference stays 64-bit |
| 32/64-bit D3D9 | Matching addon bitness | Staging surfaces; GPU composition of matched input/output |
| 32/64-bit D3D10/10.1 | Matching addon bitness | Staging textures and asynchronous helper |
| 32-bit D3D12, Phoenix only | Full `Phoenix.addon32` | Safe x64 helper until native x86 execution passes runtime validation |
| 32-bit D3D12, NVIDIA DLL or NVIDIA comparison | `Phoenix.addon32` | Pixel buffers with 256-byte row alignment to x64 helper |
| 32/64-bit OpenGL | Matching addon bitness | Pixel buffers; explicit framebuffer row and texture byte-order handling |
| 32/64-bit Vulkan | Matching addon bitness | Pixel buffers; submission respects the game's presentation wait semaphores |

D3D11 falls back to staging if shared NT textures/fences are unavailable.
DirectX 8 and earlier need a separate wrapper to a ReShade-supported API.

## Automatic processing location

For standalone input, `[Phoenix.Compatibility] ExecutionMode=auto` is the default.
`in_process` explicitly forbids helper fallback; `helper` explicitly selects the
companion. `StandaloneInput=1` still selects final-color capture independently
of the processing location. Native engine DLSS input remains a separate route.
Full x86/x64 addons advertise `CnrPHOENIXInProcessBackbufferVersion` (version 1);
thin frontends and older builds without this capability use the helper.

The initial in-process implementation supports the Native RTX executor with
single-sample, single-layer SDR RGBA8, BGRA8 and RGB10A2 backbuffers on the game's
DIRECT queue. Conversion uses FP16 textures and GPU packing, retaining 10-bit
code values on RGB10A2 rather than passing them through the helper's RGBA8 pipe.
Capture/conversion/evaluation/presentation stay on that device. Motion is zero
and depth is absent: this route does not invent engine guides. Phoenix, NVIDIA
DLL and supported pair comparisons use this same in-process renderer on x64.
On x86, the driver-validation gate currently keeps all processing in the helper;
a comparison that uses NVIDIA output also requires the x64 NVIDIA DLL. Selecting a backend
re-evaluates the route without requiring an off/on cycle. The NVIDIA
feature stays resident across backend switches at unchanged geometry and
create-time settings. Its first use still loads and initializes the runtime;
switching back does not start a new renderer or repeat feature warmup. Portable
Phoenix executors retain their isolated helper route; NVIDIA-only rendering
ignores the saved Phoenix executor without changing it. Other color spaces,
protected surfaces and unsupported layouts are refused, not silently downscaled
or reinterpreted. Only one window owns the native evaluator; other windows can
fall back to isolated helpers.

Initialization or safely abandoned evaluation failures select the helper once,
with a concrete `fallback_reason`. An unretired GPU submission stops processing:
it must never be followed by freeing its in-flight resources or starting another
route over uncertain work. The in-process route waits for completed inference
proof before recording final writeback, then retires that writeback before
counting a presented result. It currently synchronizes the CPU to GPU completion;
eliminating the companion does not imply a fully asynchronous or zero-stall path.

Disabling the in-process effect or selecting the original image retains its
initialized model and textures. Re-enable resets history without starting a
process. Geometry/model/architecture changes still rebuild the affected resources
after retirement. Compatibility helper toggles retain their existing lifecycle.

Read-only diagnostics and the control mailbox report `execution_mode`,
`execution_preference`, `helper_pid` (zero in-process), and `fallback_reason`.
The installer writes automatic selection and accepts older helper-only payloads.
The existing packaged binaries are not automatically replaced by source edits.

Focused compile/CPU verification (does not launch a game, model or GPU):

```powershell
python tools/check_backbuffer_routing.py --ngx-sdk C:/SDKs/ngx/include
```

Live verification remains explicit. Tests must require the actual route,
not count a successful helper fallback as an in-process pass:

```powershell
python tools/test_graphics_apis.py --package C:/PhoenixPackage --weight-source C:/model-source.json --reshade32 C:/ReShade32.dll --reshade64 C:/ReShade64.dll --bits 64 --apis d3d12 --execution-mode in_process --motion-checks --resize-checks --output C:/ExplicitTestOutput
```

**In-process validation on 2026-10-16 (RTX 5080, driver 616.64):** the real
installer deployed the packaged addon to the museum DX12 x64 fixture. Automatic
selection used `in_process`, with `helper_pid=0`, no spawned graphics helper and
no fallback, at 1280 x 720 and maximized 2560 x 989. Final-present PNG captures
from the same frozen scene differ from disabled captures: central RGB mean
absolute differences are 10.18 and 8.76 out of 255 respectively. This establishes
delivered pixels, not a visual-quality score. Both runs completed three off/on
cycles without reconstructing the network. Control polling observed resumed
frames within approximately 0.49 seconds; its 250 ms polling cadence is included
and this is not per-frame latency. First observed output took about 10.2 seconds
from process launch, including fixture/ReShade/model initialization. Native model
GPU intervals averaged about 2.56 ms at 720p and 4.47 ms maximized; these exclude
game rendering, conversions and CPU synchronization. Both fixtures were closed.
The compact current receipt is `artifacts/current/addon/runtime-check.json`.
HDR, RGB10 runtime conversion, resize/device loss and automatic failure fallback
were not exercised by these museum runs. Historical validation below predates
this route and is not proof of its behavior.
The CPU-only checkpoint compiled both addon architectures and the integration
fixture, passed 61 routing checks, 21 session-policy checks, 32 feeder-contract
cases, 14,537 graphics-layout checks and two conversion-shader compiles per
architecture. The installer/evidence integration also passed 43 focused tests.
The full package build also passed 18 CPU suites and two software-DX12 suites.
The working DX12 fixture now has the tested addon; the clean set is untouched.

**Normal-startup correction on 2026-10-16:** the earlier controlled mode tests
temporarily enabled processing and restored a saved `Enabled=0`. They validated
the renderers, not enabled startup. The installed fixture now saves `Enabled=1`.
A normal maximized launch at 2560 x 989 started the saved NVIDIA mode without
control overrides. Foreground desktop captures with the camera frozen showed
central RGB mean differences of 8.25/255 (NVIDIA) and 7.68/255 (Phoenix) against
disabled processing. The Input/Phoenix split matched its standalone sources with
mean errors of 0.00 and 0.20/255. These are displayed-pixel checks, not FPS or
quality claims. The fixture was closed with enabled startup preserved. The mode
test now rejects disabled startup unless explicitly requested for diagnostics.

**Interaction follow-up on 2026-10-16:** the full and compact controls now share
one enable switch. Backend/comparison choices apply without toggling processing;
expensive tuning remains pending until Apply. Backbuffer routes consistently
reset spatial history because they have no engine motion vectors. This also
removes a previously measured mismatch between the in-process Phoenix output
and a fresh helper's Phoenix half. The final automatic-route run passed all
four pair checks byte-for-byte, all four Input/Phoenix difference views, gain,
Off/select/On and rapid-selection checks. A normal visible launch confirmed
comparison labels/divider. See [interaction contract](ADDON_USABILITY.md) for
the tested scenarios and limitations. NVIDIA difference views remain unavailable.

**NVIDIA/comparison follow-up on 2026-10-16:** the updated package passes 19 CPU
suites, both software-DX12 suites, and extraction/hash verification of all 101
packaged core/companion resources. Cache discovery now resolves the actual
writable AppData location under packaged launchers, retaining extended paths and
native separators for deep kernel directories. Containment checks remain active.
The project's embedded `phoenix-nvngx.dll` is a small call wrapper compiled from
`nr_forwarder.cpp`, not NVIDIA code or a model. It provides the caller-module
identity expected by the externally supplied NR runtime. An isolated probe
measured direct initialization returning `0xBAD00002` and wrapped initialization
returning success. Feeder warm-up submits feature-creation work as an unchanged
frame, without claiming processed output. Helper comparisons now retain their
requested sources and are composed once, instead of returning Phoenix-only
output and replacing the left half during transport. The legacy three-way value
is normalized to the two-way view offered by the direct panel. Failures retain
their reason instead of becoming a generic disabled message.

The real DX12 x64 museum was rerun hidden, frozen, at 960 x 540, with both forced
helper and automatic routing. NVIDIA feature 18 initialized and evaluated
successfully; Input/Phoenix stayed in-process under automatic routing. After at
least 20 NVIDIA evaluations, Input/Phoenix, NVIDIA/Phoenix, Phoenix/NVIDIA and
Input/NVIDIA pairs matched their respective standalone captures byte-for-byte
away from the divider, placed at 37% rather than 50%. Input/Phoenix absolute
difference view also rendered. Manual camera movement invalidated an earlier
visible comparison attempt; it is not counted as a renderer failure. These are
presentation-resource readbacks from a hidden window, not visible-game FPS or
VR validation. NVIDIA-source difference views remain unsupported; this check
covers NVIDIA pair wipes, not three-way output or every difference visualization.
The fixture and helper processes were closed and its INI restored.

```powershell
python tools/test_museum_modes.py --background
python tools/test_museum_modes.py --background --helper
```

## Shared panel

The main addon and thin x86 addon call the same controls and timing-table code
in `src/integration/reshade/shared_panel.h`. Settings, presets, hotkeys, model
browsing/validation, device and CPU-thread selection, resolution/aspect/upscaling,
neural controls, network experiments and logging are shared. Runtime adapters
provide metadata and actions; the UI does not own inference libraries.

Per-frame strength, skin/mask and NVIDIA tuning changes apply live, including
while dragging, without restarting the helper. Model/executor/resource edits
reload automatically once editing ends. There is no manual Apply button.
The helper returns actual available devices, model-validation results,
processing geometry and timing rows over bounded background IPC. The larger UI
snapshot is queried at most once a second, separately from per-frame completion
checks. Rows survive helper restarts (up to 64 prior workloads per timing table).
No UI pipe operation waits on the game's render thread.

Only actual engine motion/depth and temporal-guide controls are disabled when
final-color capture cannot supply their inputs. Skin overrides and automatic
skin masking remain editable. Pair comparison supports Input, NVIDIA and Phoenix;
NVIDIA difference views remain unavailable. GPU-time adaptive
depth is available; the hidden helper's presentation rate cannot supply a
whole-game FPS controller, so that target is disabled there. These capability
limits are independent of addon bitness.

The optional UI test renders the actual widgets without a game/GPU, using
external matching **ImGui 1.92.5** sources:

```powershell
python tools/test_shared_panel.py --imgui-source C:/SDKs/imgui-1.92.5 --output build/panel_checks
```

Each bitness renders 24 combinations covering all routes, advanced controls,
CPU/manual thread selection and model-source widgets. Runtime/GPU correctness
still requires the separate bridge checks below.

**Validation on 2026-10-14 (RTX 5080 / Windows):**

- All 14 core source-test suites passed.
- Eight GPU/runtime checks passed: x86/x64 exact transport and Phoenix output,
  split display, CPU pixel transport, helper death, stale-output rejection,
  an irregular 257 × 255 canvas and 1920 × 1080 inference.
- x86, x64 and CPU-transfer Phoenix images were byte-identical.
- Real ReShade checks cover D3D9, D3D10, D3D10.1, D3D11, OpenGL and Vulkan;
  the D3D12 helper path is also exercised through the thin x64 test frontend.
  All seven return byte-identical 257 × 193 output, including readback of the
  presented pixels, with no ReShade effect shaders required.
- Both x86/x64 transport-layout suites pass 9,193 checks. Updated x86 CPU-pixel
  and D3D11 shared-GPU probes match x64 output at the small extent and 1080p.
  Helper-death and expired-output checks pass on the updated x86 sharing path.
- The packaged x86 addon loads in **Sid Meier's Pirates! / D3D9**, starts the
  model helper, completes inference and submits processed frames for display.
- Windows still denies the x86 real-ReShade synthetic harness. Its other APIs
  compile and share the checked transport code, but complete x86 real-API
  certification remains open. Security settings were not changed.

These are functional tests, not performance or visual-quality certification
for every game/API. Processing runs at presentation, before ReShade effects and
its overlay, so a dummy shader is unnecessary and ReShade menus stay outside the
neural feed.

**Current limits:**

- Windows must be 64-bit, even for a 32-bit game. The addon and ReShade in the
  game must match the game's bitness. NVIDIA's modern CUDA host toolchain is
  64-bit; the thin x86 addon keeps that code in its companion process.
- Bridges currently handle single-sample SDR RGBA8/BGRA8 presentation.
  HDR and unsupported frame formats are refused visibly, preserving the game.
- A final-color hook does not supply true motion vectors or scene depth. The
  compatibility route uses neutral guides and resets history on every frame.
  It is not equivalent to an engine-integrated temporal feed.
- Inference is asynchronous. **Matched-frame display is the default**: display
  the last completed output, with its matching input in the original half of
  comparison mode. Hold that pair until the next result is ready, including
  when the helper stops after a successful frame. New game frames are captured
  before composition, so the displayed result is never fed back into the model.
  Startup shows the original game until the first result; disabling the bridge
  or selecting Original also restores the live game. This adds processing delay
  and limits visible animation to the completed-result rate, without blocking
  the game thread on an unfinished helper or GPU fence.
- Turn off **Keep input and output from the same frame** to try D3D9/D3D11
  live-frame editing. That applies an older edit to the latest image and can
  show newer motion underneath older masks. Capture-age limits apply only in
  this mode; expired output falls back to the live game. Other APIs still use
  delayed full images. ReShade deferred lists without proven submission ordering
  are refused.
- NVIDIA feature 18 and NVIDIA pair comparisons now pass the DX12 x64 museum
  check above using the embedded call wrapper and external model DLL. Historical
  `0xBAD00002` results predate that wrapper. Other APIs/architectures, NVIDIA-source
  difference views and game-specific HDR/guide contracts are not certified by it.

## Transport latency

The helper now uses a compact in-memory completion snapshot instead of writing
and rereading the full diagnostic report for every frame. Hidden helpers wait
for the next pipe request instead of polling/presenting with an 8 ms sleep;
their interactive debug-window mode keeps its message/presentation pump.
Frame pipe buffers are 1 MiB, and the x86 embedded core extracts without a
second approximately 195 MiB allocation. Cached-file hashing streams 64 KiB chunks.

Matched single-process warm checks on RTX 5080, driver 616.64, Ryzen 9950X3D:
16 calls each, first excluded; full model, eight blocks, one pass, reset each call.
Values measure the helper round trip, excluding game capture/display and startup.

| Boundary | Previous ms | Updated ms |
| --- | ---: | ---: |
| 1080p CPU-pixel transport echo, no inference | 19.093 | 4.335 |
| 257 × 193 CPU-pixel transport plus Phoenix | 41.157 | 7.312 |
| 1080p CPU-pixel transport plus Phoenix | 47.632 | 19.447 |

All output hashes match. These are preliminary timing samples, not game FPS or
a comparison with direct D3D12 inference. The panel and status export distinguish
helper round trip from the last capture/upload/display CPU-call costs. D3D9
readback itself can still wait inside the driver. Matched-frame presentation
prevents mixed-frame masks but does not supply missing engine motion vectors.

## Install and use

1. Install addon-capable ReShade matching the game's bitness.
2. Copy **one** matching addon file beside the game executable. A 32-bit game
   gets `Phoenix.addon32`; a 64-bit game gets `Phoenix.addon64`.
3. Keep your NVIDIA model DLL separate, or select a supported external model
   folder. No NVIDIA DLL or model weights are embedded in these addon files.
4. Open ReShade's **DLSS5 Phoenix** panel, choose the model with **Browse...**,
   turn on **Enable neural rendering**. Controls apply automatically.
5. The panel reports startup/model validation, completed frames, round-trip
   latency and errors. Model/device/resource changes reload the renderer;
   per-frame tuning does not restart it.

The helper executable, its ReShade runtime and required x64 Phoenix code are
embedded. They unpack into `%LOCALAPPDATA%\DLSS5Phoenix\runtime`; private session
configuration/logs live under `%LOCALAPPDATA%\DLSS5Phoenix\sessions`. No large
runtime folder needs to be copied beside the game. The helper is hidden and
owned by a kill-on-close job; it does not survive its game process.

In Phoenix mode the external DLL supplies model data. In NVIDIA execution mode
the supplied DLL is executable vendor code, loaded only in the x64 process.
That mode therefore needs the actual compatible runtime, not just a model folder.

## Build

Use the normal [build requirements](../README.md) plus MSVC's **x86 C++ tools**
and an external addon-capable **ReShade x64 6.8.0** runtime matching the vendored
SDK/ImGui headers. The scripts select C++20. Supply your validated model-free
runtime/native bundles as for the normal embedded addon build:

```powershell
python tools/build_addon.py compatibility_core --production-only `
  --ngx-sdk "C:/SDKs/ngx/include" `
  --embed-runtime "C:/Builds/validated_runtime" `
  --embed-native "C:/Builds/validated_native"

python tools/build_compatibility.py compatibility_package `
  --core-build build/compatibility_core `
  --ngx-sdk "C:/SDKs/ngx/include" `
  --reshade64 "C:/SDKs/ReShade/ReShade64.dll"
```

Outputs are `build/compatibility_package/Phoenix.addon64` and
`build/compatibility_package/Phoenix.addon32`. The package builder checks PE
architecture, the complete helper ABI, and source/object/resource hashes from
the successful core build. Changing those inputs requires rebuilding the core;
an old binary's successful test report cannot approve different linked objects.
Both addons compile the full evaluator. The x86 build uses MinHook's x86
decoder and the system `nvapi.dll`, rather than trying to load `nvapi64.dll`.
It exports undecorated public aliases for WINAPI entry points so component
inspection and control clients can recognize the same ABI on both architectures.
Its embedded native GPU kernels are shared with the x64 build; the embedded
x64 portable plugins and NVIDIA helper remain isolated from x86 native execution.
Compilation, driver entry-point inspection and CPU route tests do not establish
live x86 inference or presented pixels. Those require a separate explicit run.
It rejects arbitrary DXGI
proxy DLLs. It embeds licenses for the MIT compatibility host and BSD ReShade
runtime. All generated outputs remain ignored by Git.

The reproducible probes are [compatibility_bridge_probe.cpp](../tests/native/compatibility_bridge_probe.cpp)
[compatibility_d3d11_game.cpp](../tests/native/compatibility_d3d11_game.cpp),
and [compatibility_d3d9_game.cpp](../tests/native/compatibility_d3d9_game.cpp).
The bridge's read-only `CnrPHOENIXCompatibilityStatus(char*, uint32_t)` export
returns required JSON-buffer bytes (including NUL), so matching-bitness hosts
can inspect status without controlling the desktop.

Run the real API matrix with independently supplied ReShade binaries and Vulkan
headers (the Vulkan manifest/environment are private to the test process):

```powershell
python tools/test_graphics_apis.py --package build/compatibility_package `
  --weight-source C:/Models/weight_source.json `
  --reshade32 C:/SDKs/ReShade/ReShade32.dll `
  --reshade64 C:/SDKs/ReShade/ReShade64.dll `
  --vulkan-headers C:/SDKs/Vulkan/include `
  --output build/graphics_api_checks --bits 32 64 `
  --apis d3d9 d3d10 d3d10_1 d3d11 d3d12 opengl vulkan --motion-checks `
  --control-checks --multiwindow-check --resize-checks
```

The harness checks actual API identity, completion, presentation and cross-API
pixel equality. `--motion-checks` changes the live image while an older frame
completes, checks that the old input/output pair stays matched, and then requires
new processed output from the changed input. `--control-checks` exercises the
game's file-control endpoint, bypass/restart, and exact screenshot pixels.
`--multiwindow-check` checks independent OpenGL windows on one shared context.
`--resize-checks` cycles D3D10/10.1, D3D11 and D3D12 DXGI canvases through
321×241, 641×361 and back to 257×193. It requires fresh processed output at
each extent and an identical output hash when returning to the original canvas.
The test releases its own graphics bindings before resizing; any addon-held
backbuffer view must also be released through the runtime reset callbacks.
Both failed receipts and failing process exit codes reject a test run. Repeat
with `--display 2` to check matching comparison halves. A Windows execution
denial is recorded rather than bypassed.

The shared UI CPU test requires external ImGui sources with the **exact docking
header matching the vendored SDK**, currently `v1.92.5-docking`. A matching
version number alone is insufficient; the tool checks header identity and
runtime structure layouts before accepting results:

```powershell
python tools/test_shared_panel.py --imgui-source C:/SDKs/imgui-v1.92.5-docking `
  --output build/shared_ui_checks
python tools/test_compatibility_ipc.py --output build/ipc_checks
```

## Control without desktop input

Compatibility sessions expose the same bounded `phoenix-control/1` file protocol
as the direct addon. The first window publishes `PhoenixOutput/bridge/endpoint.json`
beside the addon or in its explicitly configured sandbox. Additional windows
have separate `bridge-<window ID>` directories. The status export includes
`window_id` and `control_folder` for discovery.

Read the endpoint's `session` and `pid`, then atomically publish `request.json`.
Use a unique request ID and a deadline no more than 30 seconds ahead of the
same machine's monotonic tick clock. Read `response.json` with the matching ID.
For `set`, optional top-level `enabled` and `coherent_frames` booleans control the
compatibility frontend; ordinary settings retain their existing names such as
`Backend`. Direct-D3D12 sessions reject these frontend-only flags.

Supported operations are status, set, save, reset_history, screenshot, capture
and validate_model. A queued reply acknowledges work rather than proving that
a model loaded or an image was written; query status for completion. Model
validation uses a separate CPU broker and does not activate the candidate.
Screenshots capture the actual pre-Present game resource and encode only after
retirement. Unsupported HDR formats are rejected rather than read into an
undersized SDR buffer.

## Helper lifetime and memory

- Protocol version 10 uses a unique endpoint per helper session. The client
  duplicates fence handles into its own process; CPU sessions need no shared fences.
- Worker state outlives its game-facing owner safely. Shutdown requests return
  without joining on Present, and cache preparation checks cancellation.
- D3D11 retains its D3D12 sharing device for the game graphics device's lifetime.
  This avoids driver-context teardown on every canvas resize or backend reapply.
  Pixel textures, presentation views, fences and helper sessions are still
  replaced independently; the sharing-device owner is cleared on game-device
  destruction or normal addon detachment.
- Iteration scratch exists only for multi-pass routes that need it. Upload and
  readback storage is reused after proven completion; intermediate refinement
  stays on the helper GPU for RGBA8 transport. The external FP16 multi-pass CPU
  route preserves its previous half-rounding, byte-conversion and upload sequence
  to avoid changing model output.
- Presets/hotkeys persist without restarting inference. Imported FPS-budget
  profiles switch to the supported companion GPU-time budget with an explanation.
- Timing rows identify the companion GPU explicitly. Helper round trip excludes
  actual game capture/display cost. D3D9's deferred snapshot readback still uses
  a synchronous driver call; no zero-stall claim is made.

Compatibility screenshots capture the composed game/Phoenix image before
ReShade effects and UI. HDR and multisampled captures require a conversion or
resolve path and are currently rejected. Validation confirms model integrity,
loading and three finite CPU outputs; it does not certify visual quality or
compatibility with every GPU runner.

For paired old/new helper timing and sampled process-memory checks, install
`psutil` in the benchmark Python environment and run
`tools/benchmark_compatibility.py --help`. The tool alternates old/new probes,
requires exact output hashes, and labels WDDM/process peaks including startup.
Do not sum client and helper GPU counters: shared allocations may appear in both.

With a supplied external model, run the direct matrix separately from model-free builds:

```powershell
python tools/test_compatibility.py --package build/compatibility_package `
  --weight-source C:/Models/weight_source.json --full-hd
```
