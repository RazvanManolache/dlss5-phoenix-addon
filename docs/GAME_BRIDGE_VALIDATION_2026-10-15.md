# Graphics bridge validation — 15 October 2026

## D3D12 final-frame SDR input — 16 October

The compatibility bridge now accepts D3D12 RGB10A2 SDR backbuffers, in addition
to RGBA8/BGRA8. It unpacks capture pixels into the helper's RGBA8 transport and
packs completed pixels back into a same-format RGB10A2 upload texture. RGB is
rounded to 8-bit during processing; this is not a lossless 10-bit or HDR path.
PQ/HLG and other non-sRGB color spaces are refused explicitly.

For a game whose native NGX output is not reliably visible, the explicit
final-frame route is `input_provider=feeder_d3d12` in `Phoenix.json`, together
with `[Phoenix.Compatibility] StandaloneInput=1`, `Enabled=1`, and
`CoherentFrames=1` in `ReShade.ini`. `[Phoenix] Backend=3` selects Phoenix.
This uses the packaged x64 helper and holds a completed input/output pair at
presentation; it does not repair the game's native NGX integration or NVIDIA NR
feature creation. Original engine depth/motion are unavailable on this route.
RGB10A2 screenshots remain outside the compatibility screenshot encoder's
accepted formats. No new real-game visual acceptance is implied by this change.

The final package passed 16 CPU suites and two software D3D12 suites. Its shipped
x64 add-on also passed the RGB10A2 SDR GPU fixture, matching-frame motion check,
and repeated resize check (257x193, 321x241, 641x361, then 257x193). Row-layout
and packed-color conversion validation passed 14,537 checks. The presented output
matched the existing RGBA8 reference hash. Evidence is local in
`build/re9_final_frame_verified_rgb10_checks_20261016/report.json`.
The add-on and final-frame configuration were deployed with the game closed;
gameplay appearance and performance still require a separate visual check.

This checkpoint covers bridge correctness and startup behavior, not an all-games compatibility claim. The wider installer sweep is still in progress.

- D3D11 saved context states now inherit a game's single-threaded creation flag. Outcast: Second Contact previously rejected the state with E_INVALIDARG and fell back to CPU transfers. The corrected build used shared GPU textures for its 2560×1080 main menu, delivered changed finite paired captures, and passed disable/bypass/resume checks. The game and helpers were closed. Captured mean absolute RGB difference was0.02247585; changed pixels do not establish better image quality. Sampled helper round trips9.52–11.09ms exclude game capture/display.
- `Phoenix.Compatibility.StandaloneInput=1` explicitly selects the backbuffer frontend in an x64 D3D12 game. `Enabled` remains its independent on/off switch. Native game-DLSS input stays the default when this option is absent. A backbuffer installation must use `input_provider=feeder_d3d12` so game NGX calls are not enhanced a second time.
- The graphics matrix now tests the actual shipped x64 addon for D3D12. The old test-only thin frontend requires `--thin-d3d12`. The packaged addon passed base output, matching-frame motion composition, control/screenshot, and resize checks. Those are synthetic tests, not gameplay evidence.
- Command/device hooks are queued and activated together. This avoids repeated thread enumeration/freezes during startup. CPU tests cover one activation for multiple hooks, uncommitted cleanup, queue/activation failure rollback and the empty batch. Hook failure does not clear a trampoline that could still be active.
- Single-threaded D3D11 transport and inference probes passed for x86 and x64 after batching. Transport was byte-exact; both inference probes produced the same output hash. Expired results, a killed helper, matching-frame presentation and UI metadata were checked. Whole-probe durations include setup and deliberate expiry/death checks, so they are not inference timings.

The local evidence is in `build/game_install_20261015`. The first x86 transport attempt before batching timed out at startup; a debugger retry passed. Its thread trace caught hook setup inside repeated system-thread enumeration. After batching, all four sequential transport/inference checks passed (whole-probe times10.95s,10.18s,6.62s,6.54s respectively). This bounded run does not prove every startup timing condition is fixed.

A Resident Evil native-input attempt captured a black startup frame and was closed; it is not counted as a pass. Pirates needs a repeat of its earlier restart failure, and ioquake3 needs bare-hook/game baseline isolation. Keep these failures visible while improving the installer using architecture, renderer/engine evidence and parameters rather than title-specific rules.

## Vulkan presentation queues

Some Vulkan games present on a queue other than ReShade's graphics queue. The
compatibility frontend now records graphics work on the runtime graphics queue
and retires its readbacks there after ReShade flushes that work. Device and window
identity checks remain in place. The bridge endpoint is published during runtime
initialization, including before usable presentation, so startup diagnostics can
query its state. An enabled frontend waiting for its helper reports Starting.

Vulkan swapchains request transfer-source and transfer-destination usage at
creation. Reading the game image requires the former; uploading the processed
output requires the latter. Usage cannot be added after swapchain creation.

The packaged x86 and x64 builds passed the compute-only presentation queue
fixture with a separate graphics queue and a swapchain whose application request
omitted transfer-destination usage. Both produced the same expected output hash,
`cda2d23537e7e7da4ebc27afd1c51cb3938dd12dfc690bc51eaa97e5423ec8be`.
Evidence: `build/game_install_20261015/vulkan_ready_compute_matrix/report.json`.
Core validation also passed 15 CPU suites and two software D3D12 suites.

Real Quake and Quake II remaster checks with the preceding queue-fix package
returned processed Vulkan output and varied screenshots, but reported below
1 FPS. Those checks establish output delivery, not playable performance. The
inactive-window baseline and bridge cost still need isolation. Outcast's shutdown
access violation was reproduced at the same game offset with bare ReShade and
with Phoenix; an entirely unmodified baseline is still required.

## D3D9 multisampling and capture — 16 October

Narbacular Drop exposed an 8x multisampled D3D9 backbuffer. The compositor
already resolves its capture through `StretchRect`, but private staging/upload
descriptions inherited the game sample count and initialization rejected it.
Those private surfaces now use one sample. D3D9 screenshot capture follows the
same rule; other APIs retain their explicit multisample refusal.

The x86 and x64 real-ReShade fixtures now draw a known image into an 8x target,
independently resolve the actual presented result and check bridge controls and
screenshots. Both matched the single-sample reference hash above, including
byte-identical processed screenshot pixels. Evidence is in
`build/game_install_20261015/source/build/game_msaa_capture_matrix/report.json`.
The updated core passed 15 CPU and two software D3D12 suites.

The real 32-bit game subsequently delivered native RTX output. Its first
retest lacked screenshot support, which prompted the separate capture fix;
that first run is output-delivery evidence only. Native kernel binaries and
runner selection were unchanged. No general HDR, VR or gameplay-quality
acceptance follows from these tests.
