# Direct D3D12 Feeder API

The live add-on exposes `CnrPHOENIXFeederVersion()` (ABI 1) and
`CnrPHOENIXFeederEvaluate(frame, result)`. The public layout is
[`feeder_api.h`](../src/integration/reshade/feeder_api.h).

This path calls Phoenix directly, without creating/evaluating a Super Resolution
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
verify Phoenix completion/writeback before presenting the output.
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

Stream 0 is mono; 1 and 2 are left/right. Phoenix retains separate executors,
timing and history and switches only after the previous stream retires. Global
control resets propagate to every stream. Stereo accepts Input/Phoenix sources;
NVIDIA-reference sources are refused until their NR features are isolated.
Synthetic transport validation is not stereo-game acceptance or a quality claim.
