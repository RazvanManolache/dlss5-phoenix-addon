# Addon Interaction Contract

## Processing And Selection

- **Enable neural rendering** is the single processing switch. Off displays the
  original image and retains the selected backend. Selecting a backend while
  Off does not turn it on.
- **Backend** selects Input, NVIDIA DLSS5 DLL, Comparison, or Phoenix DLSS5. Input is
  an intentional model bypass, not another enable switch.
- Backend, comparison view, sources, source swap, divider, difference gain,
  labels and separator choices apply directly. No off/on cycle is required.
- Strength, skin, mask, intensity and other per-frame tuning controls apply
  immediately, including while dragging. They do not restart the renderer.
  Model, executor and resource-layout changes reload automatically once the
  editing gesture ends. There is no manual Apply step.
- Full and compact controls share the enable and selection behavior. The
  backend hotkey changes only the selection, not unrelated renderer configuration.
- Strength and skin overrides remain editable with standalone final-color
  capture. Automatic skin masking enables the skin strength slider; only
  engine motion/depth controls require actual engine guides. These tuning
  changes are live in the game process and use coalesced asynchronous messages
  for helper-backed routes. A returned helper frame is only marked current once
  it includes the latest tuning revision.
- Processing location is an advanced transport choice, not an enhancement
  switch. Automatic chooses in-process processing where supported and a helper
  otherwise. A new helper can take time to start; the game remains visible.
- Eligible x64 DX12 standalone sessions keep Phoenix DLSS5, NVIDIA DLSS5 and pair
  comparisons in the game process. NVIDIA DLSS5 initializes on first use and stays
  warm across backend switches. An unchanged feature is not recreated just
  because the backend changes. Explicit Helper mode and older graphics APIs
  still use the companion renderer.

The former compatibility controls were independent: **Enable graphics bridge**
allowed frame transport, while **Neural rendering (master switch)** could tell
the renderer to bypass the model. Compatibility now migrates these settings
into one switch and always enables the model inside an active transport.
Legacy profiles with either switch Off remain Off until explicitly enabled.

## Comparison And Errors

Pair wipe supports Input, NVIDIA DLSS5 and Phoenix DLSS5 sources, including reversed pairs.
Input/Phoenix DLSS5 also supports difference views. NVIDIA DLSS5 difference views are not
implemented; selecting a NVIDIA DLSS5 source uses pair wipe rather than accepting a
nonfunctional difference selection. Identical sources are allowed and labeled
as such. Labels and dividers describe the presented selection, not pending work.

Recoverable failures retain the requested On state, suspend processing and
show the actual error. A new selection or **Retry rendering** retries processing. An uncertain
GPU submission remains a hard stop requiring a game restart; changing controls
must not bypass this safety rule. Suspended attempts do not consume the next
selection's initialization timeout.

Backbuffer capture has no engine motion vectors, so both in-process and helper
processing use spatial history resets. A synthetic zero-motion texture is not
evidence of a stationary scene. This keeps Phoenix DLSS5 consistent across standalone
and NVIDIA DLSS5 comparison routes without changing real engine-guide integrations.

## Regression Coverage

`tools/test_shared_panel.py` drives real ImGui widgets on CPU, including both
bitnesses and full/compact views. Policy tests verify that live tuning cannot
change model/resource settings and that suspended initialization deadlines reset.

`tools/test_museum_modes.py --background` separately exercises the DX12 fixture.
It changes modes without re-enabling processing on every case, requires a fresh
presented selection revision, checks comparison pixels, tests Off/select/On and
rapid source/divider changes, then closes its own processes and restores the
starting configuration. This is bounded fixture coverage, not a claim that
every game, graphics API, HDR format or VR transport has been tested.
Automatic-route checks also reject an unexpected helper and verify that
returning to NVIDIA DLSS5 reuses its feature while reproducing the standalone pixels.

### Verified Checkpoint: 2026-10-16

Two independent reviews covered the shared controls and runtime lifecycle.
The final build passed 19 CPU suites, two software-D3D12 suites, and 399 real
widget checks per bitness (x86 and x64).

The automatic-routing DX12 run passed all nine mode cases without re-enabling
processing between selections or forcing resets through the test setting.
All four checked pair wipes matched their standalone sources byte-for-byte,
including reversed NVIDIA DLSS5/Phoenix DLSS5 sources. Four difference views, 1x/8x gain,
Off/select/On cycles and rapid source/divider changes passed separately.

A normal maximized launch started the saved Phoenix DLSS5 backend with no startup
control overrides. An unobscured desktop crop confirmed the displayed Input /
Phoenix DLSS5 labels and divider at a full client size of 2560 x 989. Desktop overlays
invalidated earlier capture attempts; those are not counted as renderer failures.
The final fixture and helpers were closed, with Phoenix DLSS5 enabled and Automatic
processing retained for the next launch. Compact evidence and the binary hash
are stored in `artifacts/current/addon/runtime-check.json`.

### In-Process NVIDIA DLSS5 Follow-Up: 2026-10-16

The subsequent routing build passed 19 CPU suites, two software-D3D12 suites,
and 125 backbuffer-route checks. All nine fixture mode cases and four interaction
scenarios passed with `helper_pid=0`. The NVIDIA DLSS5 feature was created once and
reused across NVIDIA DLSS5/Phoenix DLSS5 switches, with byte-identical NVIDIA DLSS5 return images.
All four checked pair wipes again matched their standalone sources exactly.
These checks used the saved enabled NVIDIA DLSS5 startup selection, and restored it
after closing the fixture. The current receipt describes this follow-up;
the widget and foreground-label checks above belong to the preceding checkpoint.

### Live Skin Controls Follow-Up: 2026-10-16

Skin overrides no longer inherit the engine-guide disable gate. Real-widget
tests passed 1,203 checks per bitness, including 12 strength/skin scenarios per
bitness and skin changes while the mouse remains held down. The build passed
19 CPU suites, two software-D3D12 suites, 104 policy checks and 62 IPC checks.
The Apply button was removed. Per-frame controls use a live-only transaction;
helper messages are coalesced and returned frames carry tuning-revision proof.
Model/resource edits reload after editing ends. This follow-up did not launch a
game or validate hardware inference; see `artifacts/current/addon/ui-check.json`.
The preceding GPU checkpoint retains its original binary hash separately.

### DX12 x86 Runtime Follow-Up: 2026-10-16

`python tools/test_museum_modes.py --bits 32 --background --live-tuning`
passed nine mode cases, four interaction scenarios and two live-tuning checks.
Both Phoenix DLSS5 and NVIDIA DLSS5 processing ran in the private x64 helper; the x86 addon
handled capture, controls and presentation. Four pair comparisons matched their
standalone sources byte-for-byte. Tone/structure and skin changes altered real
output pixels without changing the helper PID, and restoring defaults reproduced
each backend's baseline exactly. Requested and presented tuning revisions matched.
The fixture and its helpers were closed and the starting INI restored.

This route still uses CPU pixel transport and recreates the helper for backend
or comparison-source changes. It does not execute Phoenix DLSS5 inference in x86.
Compact evidence, module names and binary identities are retained in
`artifacts/current/addon/dx12-x86-check.json`; local paths and logs are excluded.
