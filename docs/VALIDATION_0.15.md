# Phoenix DLSS5 0.15 validation

This release refreshes the shared overlay and model/resource lifetime handling. It does not claim new kernel speed or new physical-device qualification.

## Shared overlay

The CPU ImGui fixture checks the exact shared widget sources and interaction tests on both x86 and x64:

- 1,869 real-widget checks, 1,179 dashboard checks and 36 policy/action checks per architecture.
- 266 dashboard frames per architecture, at widths 360, 700 and 1,100 pixels.
- All four tabs, Save, scrolling with a fixed footer, model/setup quick navigation and a newly recreated context each frame.
- Output selections, comparison sources, enable toggles, appearance and skin/mask controls, legacy capability gates and passive drawing without settings mutation.
- Exact restoration of ImGui style/color state. Compact UI preserves the host theme.
- All 136 persisted settings and the prior literal widget declarations remain accounted for, including intentional label aliases and consolidated pass controls.

The public screenshots are CPU rasterizations of real panel draw data with synthetic status, not retail game captures. No personal desktop, camera or game data is used.

## Rebuilt release packages

- Standard and experimental-compatibility add-ons are rebuilt from current source, for x86 and x64, together with the graphics helper and forwarding shim.
- Each channel runs the normal 24 CPU contract suites and 2 software D3D12 suites. This includes model-switch lifetime and balanced NVAPI lifetime regressions.
- Actual final DLLs are loaded by architecture-matched minimal ReShade hosts. Registration, exported ABI versions, explicit initialization and balanced unload are checked; these hosts do not render games.
- Final embedded resources are decompressed, hash checked and audited recursively, including the x86 package's nested current x64 addon. The embedded helper matches the standalone helper.
- Both architectures exercise resource extraction, lazy vendor dependencies, companion extraction and cache reopening.
- The source archive equals the complete audited tracked tree. Compiled outputs, model DLLs/weights, captures, logs and private paths are excluded from source publication.
- Published release assets are checked against SHA-256 checksums after anonymous download.

Machine-readable `VALIDATION.json` binds these results to the final binary hashes, retained payload hashes and source publication commit. Passing compilation alone is not counted as inference validation.

## Optional Phoenix Input companion

The separately packaged input addon is built from the published source for x86 and x64. Minimal ReShade hosts check real DLL startup, identity and callback exports, a live status request, rejection of a stale session, explicit shutdown and balanced detach. No game, input macro or GPU inference is started by these checks.

Background gameplay, virtual focus, DirectInput, Raw Input and XInput behavior across real titles are not newly qualified by this smoke test. The companion remains optional and experimental.

## Retained evidence and limits

Portable runtimes and native kernel payloads retain their verified 0.14 byte identities, except the forwarding shim rebuilt from maintained source. Older-card kernel compilation is not physical-device validation. Earlier timing charts retain their original measurement dates.

The model-switch fixes also have earlier targeted Museum-fixture evidence recorded during development: model transitions, resource generations, balanced NVAPI lifetime and output completion. That evidence is separate from this publication's CPU/software checks and does not qualify every retail game's presentation or frame rate.

No new retail game run, hardware inference benchmark, AMD/Intel device qualification, VR comfort check or broad Input-companion gameplay test is claimed for 0.15.
