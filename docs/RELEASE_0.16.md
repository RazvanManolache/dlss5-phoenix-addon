# Phoenix DLSS5 0.16

**Catch the same pixel twice.** The addon can now save matching original and modified pictures without toggling the effect or rerunning the model.

- **Play → Save original + modified:** two PNGs plus a shared-frame receipt in a new capture folder.
- **Setup → Hotkeys:** bind the new action to a keyboard key. It starts unbound; existing shortcuts stay in their original slots.
- Pending, saved-folder and failure feedback in the shared x86/x64 UI.
- Native D3D12, in-process D3D12, shared D3D11 and staging/readback bridge integration. Helper results are matched to their accepted input, even when processing is delayed.
- On-demand capture storage; no screenshot buffers or PNG work when capture is idle.

The [capture guide](PAIRED_SCREENSHOTS.md) explains hook placement, comparison views, SDR/HDR conversion and route limits. These are matching **pipeline** images, before later game/ReShade effects and the overlay. Direct native asynchronous offload does not offer this action because its source association is not guaranteed.

## Downloads

| Download | Contents |
| --- | --- |
| `DLSS5-Phoenix-0.16-Windows.zip` | Updated x86/x64 addons and graphics helper; retained RTX 50 kernel images |
| `DLSS5-Phoenix-0.16-RTX-Compatibility-Windows.zip` | Alternative experimental x86/x64 package with retained RTX 20/30/40/50 images; older physical cards remain unvalidated |
| `DLSS5-Phoenix-0.16-Input-Windows.zip` | Separate optional background-input companion, unchanged binaries from 0.15, with instructions and licenses |
| `DLSS5-Phoenix-0.16-Portable-Runtimes.zip` | Optional runtime companions, unchanged from 0.15 |
| `DLSS5-Phoenix-0.16-Source.zip` | Audited source, tests, documentation, UI screenshots and notices |

Close the game, choose **one** main Windows package, and replace the addon matching its architecture plus `phoenix_graphics_host.exe`. Keep your settings and separately supplied compatible model. Use [ReShade with full addon support](https://reshade.me/). The Input companion is optional and is not required for screenshots or neural rendering.

The packaged helper and its nested x64 addon are rebuilt too. Portable runtimes and native kernels are retained byte for byte; this release does not claim a new inference-speed improvement. Existing x86 NVIDIA execution restrictions remain unchanged.

Fresh validation covers CPU contracts, software D3D12 capture, shared UI interactions, real final-DLL load/unload, resource extraction and recursive package audits. No retail game or hardware inference benchmark is launched for this publication. See [validation scope](VALIDATION_0.16.md).

No NVIDIA model DLLs, weights, private configurations or game captures are included. All downloads include file inventories and release SHA-256 checksums.

[Screenshot guide](PAIRED_SCREENSHOTS.md) · [Overlay guide](overlay-ui.md) · [Phoenix Input](PHOENIX_INPUT.md) · [Support](SUPPORT.md) · [Credits](UPSTREAM_CREDITS.md)
