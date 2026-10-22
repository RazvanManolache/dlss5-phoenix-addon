# Save original + modified

One button, two PNGs, one processed frame. No frantic toggling while your subject walks out of shot.

1. Enable neural rendering and wait for processed output.
2. In **Play**, use **Save original + modified** in the Compare card. It is visible even when the detailed comparison controls are collapsed.
3. The status changes from pending to saved and shows the output folder, or explains a failure.

For a keyboard shortcut, open **Setup → Hotkeys → Save original + modified**, click **Unbound** and press a key. Existing shortcut slots keep their meanings; the new slot starts unbound. The existing duplicate/reserved-key rules apply. Binding changes are persisted, and Save configuration remains available at the top.

## Saved files

Each capture gets a new `pair-…` directory inside the addon's configured `PhoenixOutput` directory:

- `original.png`: the matching processing input.
- `modified.png`: its processed result/current modified view.
- `pair.json`: shared frame identity, dimensions, capture scope and conversion information. This file is published last, after both PNGs are complete.

Existing captures are not overwritten. Screenshots are opaque SDR PNGs: source RGB is retained at PNG precision, and auxiliary/zero engine alpha is made opaque. Floating-point linear images are clipped to SDR and sRGB encoded using the same conversion for both images. These are not lossless HDR exports.

## Where the images come from

| Route | Pair source |
| --- | --- |
| Native game D3D12 / native RTX or NVIDIA output | The same owned command recording, before and after processing; the current view can include split/difference |
| In-process final-frame D3D12 bridge | Its original capture and completed presentation texture |
| D3D11 shared helper | The held, matching input/output textures from a completed helper result |
| Staging/readback bridges, including portable/native helper transports | The next accepted input and the result with that exact frame token; older results are ignored |

Helper captures save the full processed result before local comparison furniture or an optional live-background edit. They deliberately use the completed input rather than a newer game frame. Native asynchronous direct-game offload is unavailable for this action; use native RTX or a final-frame bridge when an exact pair is required.

These are pipeline captures, before later game effects, ReShade effects, labels and the overlay. Depending on the game hook, they may differ from the final displayed screenshot. To capture the entire displayed view, use the ordinary ReShade screenshot action.

Capturing does not rerun inference. Readbacks are allocated on demand and released after completion; staging helpers retain an input only while a requested pair needs it. PNG encoding and any necessary readback can briefly pause a frame. One request is active at a time. Changing a compatibility route or resolution cancels an unfinished request instead of labeling an unrelated frame as its result.

## Validation

The targeted software-rendering test mutates the **same texture** between before/after copies, decodes both resulting PNGs, and checks the distinct expected pixels and shared frame ID. It also checks fence/submission gating, discarded command lists, repeated captures, stale helper results, alpha handling and readback release. Shared ImGui interaction tests exercise the real button, disabled state and seventh hotkey slot on x86/x64.

See [0.16 validation](VALIDATION_0.16.md) for the release scope. Software tests do not establish every game's hook placement, HDR appearance or gameplay compatibility.
