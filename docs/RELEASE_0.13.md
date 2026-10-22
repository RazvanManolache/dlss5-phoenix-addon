# Phoenix DLSS5 0.13

Your pixels, your processing chain. This release brings the new chain controls
and memory-conscious model loading into the add-on packages, with matching
runtime companions rather than new controls sitting on old binaries.

## Highlights

- Reorder neural rendering, scaling, color treatment and experimental multiscale
  fusion, with independent histories for neural passes/branches.
- Optional NVIDIA optical flow, engine-vector priority, confidence rejection and
  scene-change resets on eligible final-frame routes.
- Native NIS and an optional experimental tiny 2x scaler. The tiny scaler needs
  external weights and currently has visible artifacts/high cost; it is not the
  Lightweight DLSS5 model. VSR and OptiScaler remain deferred.
- Optional pipeline monitoring, coordinated age budgets, recovery to working
  settings, comparison zoom/loupe and screenshot settings receipts.
- Updated native SDK and portable pipeline companions, native file/framed-stream
  processing, appearance forwarding and actual processing-host/neural-worker IDs.
- Streamed model hashing/selected-resource loading, bounded authenticated native
  weight staging and safer x86 address-space preflight. Retryable routes remain
  available; automatic portable device selection prefers qualified hardware.
- Original and Lightweight, Phoenix DLSS5 and NVIDIA DLSS5 comparisons, x86/x64
  add-ons, the retained x64 helper and experimental older-RTX kernels remain.

## Choose and install

| Download | Contents |
| --- | --- |
| `DLSS5-Phoenix-0.13-Windows.zip` | x86/x64 add-ons and graphics helper; retained validated RTX 50 kernel images |
| `DLSS5-Phoenix-0.13-RTX-Compatibility-Windows.zip` | Alternative experimental x86/x64 package with `sm_75/86/89/120` kernels; older physical cards remain unvalidated |
| `DLSS5-Phoenix-0.13-Portable-Runtimes.zip` | Updated architecture-matched runtime companions and native x64 CLI; backend dependencies retained from the verified preceding package |
| `DLSS5-Phoenix-0.13-Source.zip` | Audited source, tests, documentation and licenses, without models or compiled kernels |

1. Close the game. Install a current [ReShade build with full add-on support](https://reshade.me/).
2. Choose **one** Windows package. Replace the existing Phoenix add-on; do not
   load standard and compatibility variants together.
3. Copy `Phoenix.addon32` for a 32-bit game or `Phoenix.addon64` for a 64-bit game
   beside ReShade. Put `phoenix_graphics_host.exe` there too for helper fallback.
4. Keep your separately obtained model and existing configuration. Select its
   compatible DLL/native-folder path in the Phoenix panel. No model, NVIDIA DLL,
   Python, compiler or CUDA toolkit is included or needed from this source tree
   to run the packaged add-on.
5. For optional portable execution, extract `PhoenixRuntime` beside the add-on
   and select its matching architecture/runtime location. Do not merge in old
   runtime DLLs. Profile-specific experimental WOW64 packages are **not included**;
   those still need a matching locally verified Windows/driver profile.
6. Start with Automatic transport and a neutral chain. Enable optional stages
   individually; route capability limits are explained in the panel and
   [pipeline guide](PROCESSING_CHAIN.md).

## Validation boundary

This release is **build-, CPU- and software-rendering-checked only**. At the user's
request, no fresh hardware GPU inference, performance benchmark or game run was
performed for these binaries. Earlier GPU/pipeline results remain tied to their
original build identities. Compilation does not certify a game, older RTX card or
NVIDIA DLL route. See [validation scope](VALIDATION_0.13.md) and the downloadable
`VALIDATION.json` for the exact checks.

No model weights, NVIDIA model DLLs, logs, private paths/configuration or game
captures are shipped. File inventories and SHA-256 checksums accompany the assets.

[Controls and costs](PROCESSING_CHAIN.md) · [Execution choices](EXECUTION_MECHANISMS.md)
· [Support matrix](SUPPORT.md) · [Credits](UPSTREAM_CREDITS.md)
