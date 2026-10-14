# Graphics API and 32-bit game compatibility

The normal x64 D3D12 addon continues to process directly in the game. The
compatibility addon bridges **every ReShade API** to a private **64-bit D3D12 helper**:
D3D9, D3D10/10.1, D3D11, D3D12, OpenGL and Vulkan. The x86 addon uses the helper
for every API, including D3D12.
The helper uses the same Phoenix implementation, rather than a second model runner.

| Game | Addon file | Frame transport |
| --- | --- | --- |
| 64-bit D3D12 | `Phoenix.addon64` | Existing direct route; no compatibility helper |
| 64-bit D3D11 | `Phoenix.addon64` | Shared GPU textures and fences |
| 32-bit D3D11 | `Phoenix.addon32` | Shared GPU textures and fences; inference stays 64-bit |
| 32/64-bit D3D9 | Matching addon bitness | Staging surfaces; GPU composition of matched input/output |
| 32/64-bit D3D10/10.1 | Matching addon bitness | Staging textures and asynchronous helper |
| 32-bit D3D12 | `Phoenix.addon32` | Pixel buffers with 256-byte row alignment |
| 32/64-bit OpenGL | Matching addon bitness | Pixel buffers; explicit framebuffer row and texture byte-order handling |
| 32/64-bit Vulkan | Matching addon bitness | Pixel buffers; submission respects the game's presentation wait semaphores |

D3D11 falls back to staging if shared NT textures/fences are unavailable.
DirectX 8 and earlier need a separate wrapper to a ReShade-supported API.

## Shared panel

The main addon and thin x86 addon call the same controls and timing-table code
in `src/integration/reshade/shared_panel.h`. Settings, presets, hotkeys, model
browsing/validation, device and CPU-thread selection, resolution/aspect/upscaling,
neural controls, network experiments and logging are shared. Runtime adapters
provide metadata and actions; the UI does not own inference libraries.

The bridge applies an edited profile with **Save configuration / apply** at the
top. Its helper returns actual available devices, model-validation results,
processing geometry and timing rows over bounded background IPC. The larger UI
snapshot is queried at most once a second, separately from per-frame completion
checks. Rows survive helper restarts (up to 64 prior workloads per timing table).
No UI pipe operation waits on the game's render thread.

Engine-guide/history controls and NVIDIA/difference comparison views remain
visible but disabled where the final-color bridge cannot supply their inputs.
The bridge's comparison is a matching original/processed split. GPU-time adaptive
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
- NVIDIA DLL execution in the helper remains experimental. Local tests reach
  the runtime but currently fail NGX initialization with `0xBAD00002`
  (`PlatformError`). Phoenix is the verified compatibility route.

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
   enable **graphics bridge**, and use **Save configuration / apply** at the top.
5. The panel reports startup/model validation, completed frames, round-trip
   latency and errors. Model/device/settings changes restart the helper.

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
  --reshade64 "C:/SDKs/ReShade/ReShade64.dll"
```

Outputs are `build/compatibility_package/Phoenix.addon64` and
`build/compatibility_package/Phoenix.addon32`. The package builder checks PE
architecture, the complete helper ABI, and source/object/resource hashes from
the successful core build. Changing those inputs requires rebuilding the core;
an old binary's successful test report cannot approve different linked objects.
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
