# Phoenix DLSS5 0.18

**Old APIs, fewer surprises.**

- Native D3D9 readback uses system-memory surfaces, with matching compositor validation.
- SDR RGB10A2 capture and writeback cover D3D10/11 as well as D3D12. HDR capture remains guarded.
- Temporary renderer probes retain hook lifetimes safely. A later ReShade provider gets its own registrations instead of stale cached exports; stale unload notifications cannot detach the active provider.
- Process shutdown avoids teardown waits that cannot finish under the loader lock.
- Streamline resource tagging checks the observed API before treating handles as D3D12 objects.
- Optional Phoenix Input supports legacy DirectInput 7/8 A/W, buffered peek/count/discard polling, late-loaded input libraries and existing-window discovery when loaded after renderer startup.
- The control mailbox accepts paired screenshot requests. The graphics fixture accepts combined scenario and process-exit flags.

## Downloads

| Download | Contents |
| --- | --- |
| `DLSS5-Phoenix-0.18-Windows.zip` | Main x86/x64 addons and x64 graphics helper; retained RTX 50 native kernels |
| `DLSS5-Phoenix-0.18-RTX-Compatibility-Windows.zip` | Experimental x86/x64 package with retained RTX 20/30/40/50 kernel images; older physical GPUs remain unvalidated |
| `DLSS5-Phoenix-0.18-Input-Windows.zip` | Optional Phoenix Input companion for both architectures |
| `DLSS5-Phoenix-0.18-Portable-Runtimes.zip` | Retained optional portable companions |
| `DLSS5-Phoenix-0.18-Source.zip` | Audited source, tests, documentation and notices |

Close the game. Choose **one** main Windows package and replace the addon matching the game's architecture plus `phoenix_graphics_host.exe`. Keep your configuration and separately obtained compatible model. Replace Phoenix Input separately if you use it; rendering does not require it. Use [ReShade with full addon support](https://reshade.me/).

Validation for this publication is CPU and software-device only. No new game, hardware-GPU inference, image-quality or performance result is claimed. Native kernels and portable inference companions retain their published identities. See [validation scope](VALIDATION_0.18.md).

No NVIDIA model DLLs, extracted weights, private configurations or diagnostic logs are included. Packages contain file inventories, licenses and SHA-256 checksums.

[Overlay guide](overlay-ui.md) | [Screenshot pairs](PAIRED_SCREENSHOTS.md) | [Phoenix Input](PHOENIX_INPUT.md) | [Execution mechanisms](EXECUTION_MECHANISMS.md) | [Credits](UPSTREAM_CREDITS.md)
