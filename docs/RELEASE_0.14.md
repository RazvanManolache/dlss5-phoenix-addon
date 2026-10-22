# Phoenix DLSS5 0.14

A smaller release with a less exciting goal: stop choosing an unsafe route,
and explain why a route cannot start instead of waiting forever.

## Stability fixes

- **Safer NVIDIA DLSS5 in 32-bit games.** Automatic now uses the external x64
  helper for NVIDIA DLL evaluation and comparisons involving NVIDIA DLSS5.
  Repeated testing found allocation failures and crashes in the experimental
  NVIDIA-x64-inside-x86 path, so that combination is no longer offered.
- **Phoenix DLSS5 in-process support stays.** Eligible Phoenix-only x86 library
  and driver-host routes remain available with their existing model, Windows,
  driver and runtime-package requirements. x64 NVIDIA DLSS5 is unaffected.
- **An explanation, not a silent switch.** An explicitly saved incompatible
  x86 NVIDIA library selection is refused with advice to choose Automatic or
  External x64 helper. Saved preferences are not silently rewritten.
- **Startup failures stop waiting.** Once runtime discovery finishes, an
  unavailable transport becomes a retryable failure. Discovery itself remains
  asynchronous; its pending state is not treated as a terminal error.
- **More useful refusal details.** The route selector forwards missing-runtime,
  incompatible-device and low-address-space reasons. The retained experimental
  native loader source also reports failed allocation requests, Windows/NT
  errors and low-address-space context; it is not a newly supported NVIDIA route.
- Added regression cases and corrected README/transport documentation. Historical
  probe timings no longer imply qualification of the withdrawn NVIDIA experiment.

There are no new models or claimed inference-speed improvements. Runtime
companions and native kernel payloads are retained byte for byte from 0.13;
the add-ons and graphics helper are rebuilt from current source.

## Choose and install

| Download | Contents |
| --- | --- |
| `DLSS5-Phoenix-0.14-Windows.zip` | Rebuilt x86/x64 add-ons and graphics helper; retained single-target RTX 50 native kernels |
| `DLSS5-Phoenix-0.14-RTX-Compatibility-Windows.zip` | Alternative experimental x86/x64 package with retained `sm_75/86/89/120` images; older physical cards remain unvalidated |
| `DLSS5-Phoenix-0.14-Portable-Runtimes.zip` | Architecture-matched runtime companions and native x64 CLI, unchanged from the verified 0.13 payload |
| `DLSS5-Phoenix-0.14-Source.zip` | Audited source, tests, documentation and licenses, without models or compiled kernels |

1. Close the game. Use a current [ReShade build with full add-on support](https://reshade.me/).
2. Choose **one** Windows package. Replace the existing Phoenix add-on; do not
   load standard and compatibility variants together.
3. Copy `Phoenix.addon32` for a 32-bit game or `Phoenix.addon64` for a 64-bit game
   beside ReShade. Place `phoenix_graphics_host.exe` there for helper fallback.
4. Keep your existing configuration and separately obtained compatible model.
   In x86 games, choose **Automatic** or **External x64 helper** for NVIDIA DLSS5
   or NVIDIA comparisons. Existing incompatible forced selections explain this
   requirement rather than pretending to be running.
5. For optional portable execution, extract `PhoenixRuntime` beside the add-on
   and select the matching architecture/runtime location. These runtime files
   are unchanged, so an intact 0.13 installation does not need replacement.
   Profile-specific WOW64 packages are **not included**; native x86 experiments
   still require their separately verified matching bootstrap/runtime profile.

Processing chains, optical flow, NIS, optional tiny scaling and model controls
retain their [existing route limits](PROCESSING_CHAIN.md). VSR and OptiScaler
remain deferred. The package needs no Python, compiler or CUDA toolkit to run.

## Validation boundary

Fresh validation is **CPU/software-rendering-only**: rebuilt add-ons/helper,
routing contracts, add-on load/unload, extraction, embedded-content audits and
package hashes. No games, hardware GPU inference or benchmarks are run for this
release. Prior runtime/model checks stay tied to the unchanged binaries and
their earlier receipts; they are not relabeled as new 0.14 inference results.

No NVIDIA model DLLs, weights, logs, private configuration or game captures are
included. Public assets have file inventories and SHA-256 checksums.

[Validation scope](VALIDATION_0.14.md) · [Execution choices](EXECUTION_MECHANISMS.md)
· [Controls](PROCESSING_CHAIN.md) · [Support](SUPPORT.md) · [Credits](UPSTREAM_CREDITS.md)
