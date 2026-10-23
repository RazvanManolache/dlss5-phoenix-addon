# Phoenix DLSS5 0.17

**Less drama at startup. Fewer souvenirs in memory.**

- **Fallback means fallback:** an unavailable in-process library, native driver or portable runner now honours its enabled fallback setting even when a startup availability check refuses it. The normal recovery order still tries eligible same-process routes before the external x64 helper. Turning fallback off retains strict refusal.
- **Phoenix Input cleans up after itself:** unloading releases its owned hook-library heap and module references. Hook activation is batched during startup.
- **Mailboxes withstand brief file locks:** pending input requests and replies survive sharing delays without replaying completed operations. Invalid macro batches roll back only their own newly scheduled work.
- **Experimental x86 driver source:** weight upload staging is bounded to 4 MiB instead of a temporary allocation of approximately 141 MiB, with better allocation diagnostics. Profile-specific WOW64 binaries remain separately built development dependencies; this source repair does not add a machine-specific runtime to the public downloads.

## Downloads

| Download | Contents |
| --- | --- |
| `DLSS5-Phoenix-0.17-Windows.zip` | Updated x86/x64 addons and x64 graphics helper; retained RTX 50 native kernels |
| `DLSS5-Phoenix-0.17-RTX-Compatibility-Windows.zip` | Updated experimental x86/x64 package with retained RTX 20/30/40/50 images; older physical GPUs remain unvalidated |
| `DLSS5-Phoenix-0.17-Input-Windows.zip` | Repaired optional Phoenix Input companion for both architectures |
| `DLSS5-Phoenix-0.17-Portable-Runtimes.zip` | Optional portable companions, retained from 0.16 |
| `DLSS5-Phoenix-0.17-Source.zip` | Audited source, tests, documentation, screenshots and notices |

Close the game, choose **one** main Windows package, and replace the addon matching the game's architecture plus `phoenix_graphics_host.exe`. Keep your configuration and separately supplied compatible model. Replace Phoenix Input separately if you use it. Use [ReShade with full addon support](https://reshade.me/).

The standard x86 candidate passed all three formerly blocked WARP fallback cases: a real external x64 helper, continuing presented inference, changed central pixels and clean shutdown. Input lifetime and sharing-delay checks passed on both architectures. The driver staging repair passed seven targeted development-runtime checks; four API comparisons produced byte-identical output. These are generated-fixture results, not retail-game or older-GPU certification.

The preceding full 0.16 baseline covered 849 integration entries; its strict forced-route failures remain recorded as baseline failures. It is not relabelled as a full 0.17 run. Fresh final-package checks and their exact binary identities are described in [validation scope](VALIDATION_0.17.md).

No NVIDIA model DLLs, extracted weights, private configurations or game captures are included. Downloads include file inventories and SHA-256 checksums. Native kernels and portable inference algorithms are unchanged; this release makes no new speed or vendor-parity claim.

[Overlay guide](overlay-ui.md) · [Screenshot pairs](PAIRED_SCREENSHOTS.md) · [Phoenix Input](PHOENIX_INPUT.md) · [Execution mechanisms](EXECUTION_MECHANISMS.md) · [Credits](UPSTREAM_CREDITS.md)
