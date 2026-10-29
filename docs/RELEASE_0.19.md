# Phoenix DLSS5 0.19

**A steadier picture. Labels optional.**

- Hold the last completed D3D12 result during processing delays. A cold-start loading gate keeps unfinished input from flashing through.
- Restore descriptor bindings after optical-flow preparation so the feeder can keep its working native route.
- Capture engine motion at the successful game-input boundary, with ordered submission snapshots, queue/state checks and reader leases. Automatic mode retains NVIDIA optical-flow fallback when engine guides are unavailable.
- Resolve Original/Lightweight disk-model switches through the installed model library rather than reinterpreting one family's weights as another.
- **Play → Compare → Show on-screen labels:** hide source labels in full-screen or comparison views without restarting processing or clearing temporal history. Separator lines remain independent.

## Downloads

| Download | Contents |
| --- | --- |
| `DLSS5-Phoenix-0.19-Windows.zip` | Main x86/x64 addons and x64 graphics helper; retained RTX 50 native kernels |
| `DLSS5-Phoenix-0.19-RTX-Compatibility-Windows.zip` | Experimental x86/x64 package with retained RTX 20/30/40/50 kernel images; older physical GPUs remain unvalidated |
| `DLSS5-Phoenix-0.19-Input-Windows.zip` | Optional Phoenix Input companion for both architectures |
| `DLSS5-Phoenix-0.19-Portable-Runtimes.zip` | Retained optional portable companions |
| `DLSS5-Phoenix-0.19-Source.zip` | Audited source, tests, documentation and notices |

Close the game. Choose **one** main Windows package and replace the addon matching the game's architecture plus `phoenix_graphics_host.exe`. Keep your configuration and separately obtained compatible model. Replace Phoenix Input separately if you use it; rendering does not require it. Use [ReShade with full addon support](https://reshade.me/).

The release packages pass fresh CPU, shared-UI and software D3D12 checks. Earlier targeted Bodycam checks cover both models with engine-motion snapshots and NVIDIA optical-flow fallback; they are not a full game matrix or a controlled performance benchmark. Native kernel images and portable inference companions retain their 0.18 identities. See [validation scope](VALIDATION_0.19.md).

No NVIDIA model DLLs, extracted weights, private configurations or diagnostic logs are included. Packages contain file inventories, licenses and SHA-256 checksums.

[Overlay guide](overlay-ui.md) | [Screenshot pairs](PAIRED_SCREENSHOTS.md) | [Phoenix Input](PHOENIX_INPUT.md) | [Execution mechanisms](EXECUTION_MECHANISMS.md) | [Credits](UPSTREAM_CREDITS.md)
