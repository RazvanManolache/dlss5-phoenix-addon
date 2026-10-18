# Direct D3D12 Feeder API

The full x64 addon also exports `CnrPHOENIXInProcessBackbufferVersion()` (version
1). This identifies the standalone same-device frontend, not merely the presence
of an evaluator. `[Phoenix.Compatibility] StandaloneInput=1` selects final-color
input; `ExecutionMode=auto` chooses same-device Native RTX processing when its
layout/device/queue capabilities allow it, otherwise the packaged helper.
`in_process` forbids helper fallback and `helper` explicitly selects it. See
[graphics routing](GRAPHICS_COMPATIBILITY.md#automatic-processing-location) for
the limits and validation boundary. This capability does not require or create
a game SR/DLAA feature. An installed older addon is not upgraded by changing an
INI entry.

The live add-on exposes `CnrPHOENIXFeederVersion()` (ABI 1) and
`CnrPHOENIXFeederEvaluate(frame, result)`. The public layout is
[`feeder_api.h`](../src/integration/reshade/feeder_api.h).

This path calls Phoenix DLSS5 directly, without creating/evaluating a Super Resolution
or DLAA feature. Configure `input_provider=feeder_d3d12` in `Phoenix.json`.
Actual game-DLSS interception remains `game_dlss` and is unchanged.

The caller seeds the output with the original frame, provides native-size RGBA16F
color, supported motion and optional scalar R32_FLOAT depth, and records fresh
COMMON-state transitions on a fully observed DIRECT command list. Explicitly
set known descriptor-heap bindings, even when the list needs no heaps. Queue
identity must match this exact list's observed submissions; all textures must
belong to that device. The add-on recognizes ReShade's documented device proxy.
Textures must not alias. These objects are borrowed, not transferred.

`S_OK/Recorded` means commands were recorded, not that the GPU completed. The
caller must submit the same list on the declared queue, wait for its fence and
verify Phoenix DLSS5 completion/writeback before presenting the output.
`S_OK/Bypass` preserves the input. `S_FALSE/Deferred` carries no claimed neural
recording; a claimed recording that fails is terminal and must be abandoned.
There is no implicit SR fallback.

The optional `CnrPHOENIXFeederProgressJson(char*, uint32_t)` export returns a
compact read-only completion snapshot; the returned buffer size includes NUL.
It uses the same retired-frame, routing and worker counters as the full report,
without writing files or collecting the full diagnostic history on every frame.
GPU completion must still be proven by the caller's submission fence. Existing
`CnrPHOENIXSummary(path)` diagnostics remain available.

The compatibility frontend uses optional `CnrPHOENIXFeederUiJson(char*, uint32_t)`
for device, validation, geometry and timing metadata, and
`CnrPHOENIXFeederUiAction(uint32_t action, uint32_t argument)` for bounded runtime
actions. Query metadata at most once per second from a background worker; it is
larger than the frame-completion snapshot. Action values are declared in
`panel_contract.h`. No process-local UI or inference objects cross this boundary.

`CnrPHOENIXFeederLiveSettings(const char* json, uint32_t bytes)` accepts a
nonempty JSON object of at most 8192 bytes containing only the per-frame keys
listed by `visit_live_tuning` in `routing.h`. It returns BOOL acceptance, applies
valid controls atomically under the evaluation lock, and rejects model paths,
executor/geometry/create-time settings, unknown fields and invalid values without
changing the current controls. No model reload, feature recreation, disk write
or helper restart is performed. The next processed frame uses the new values.
Compatibility IPC v11 coalesces these updates on its existing pipe worker; the
host still accepts v10 clients using the unchanged frame-packet layout.

Stream 0 is mono; 1 and 2 are left/right. Phoenix DLSS5 retains separate executors,
timing and history and switches only after the previous stream retires. Global
control resets propagate to every stream. Stereo accepts Input/Phoenix DLSS5 sources;
NVIDIA-reference sources are refused until their NR features are isolated.
Synthetic transport validation is not stereo-game acceptance or a quality claim.

## Experimental appearance feeder V2

The candidate adds `CnrPHOENIXFeederVersionV2` (returns 2) and
`CnrPHOENIXFeederEvaluateV2`. The existing V1 exports and structures stay
unchanged. Construct `PhoenixFeederFrameV2`; its `frame` member is an ordinary
V1 frame, and the result remains `PhoenixFeederResultV1`. The outer size/version
identify the additive contract. V2 itself explicitly opts into the experimental
Phoenix RGB/1 appearance semantics documented in [LIBRARY.md](LIBRARY.md).

V2 supplies intensity/tone/structure/skin/style/automatic controls and optional
full-extent mask, premultiplied UI, UIAlpha, backbuffer and distortion textures.
All enter/leave COMMON and require fresh observed barriers on the supplied list.
Mask/UI/field are RGBA32F, UIAlpha R32F, backbuffer RGBA16F. UI and UIAlpha are
alternatives. Resources must be distinct, on the same device, with matching
source/network/output dimensions. The host must produce finite values in the
declared range and retain textures through completion; Phoenix additionally
holds resource references through its fence. Unsupported HDR UI/geometry rejects.

`appearance_generation` is a producer contract/coordinate/settings epoch. Change
it when those conventions change; the native/worker route resets history for
that change. Keep it stable for ordinary moving-object map updates, which must
preserve temporal continuation. V1 stream/eye and queue/fence rules still apply.
The shared panel reports producer/map/UI/field/revision; it cannot synthesize
object IDs from the final color image. See `examples/engine_control_mask.hlsl`
for an engine ID-table map producer. Addon compilation, API tests and software
D3D12 wrapper/capture checks passed; no live engine/game V2 producer was tested.
