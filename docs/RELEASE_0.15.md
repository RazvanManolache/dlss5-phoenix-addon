# Phoenix DLSS5 0.15

**The controls got a makeover. Your sliders survived.**

- **Play, Image, Performance and Setup:** everyday controls up front, detailed controls expandable.
- Colored cards, route/model status, compact timing summaries, quick links and a Save button that stays visible.
- One shared panel for 32-bit, 64-bit, native and compatibility execution. All 136 persisted settings remain represented.
- Safer model replacement: wait for outstanding work, retire unused resources, and preserve the previous model if loading its replacement fails.
- **Optional Phoenix Input companion**, packaged separately: background keyboard, mouse and virtual-controller input for local automation, without taking over the desktop cursor.

![Play: source, model and live appearance controls](https://raw.githubusercontent.com/RazvanManolache/dlss5-phoenix-addon/0.15/docs/features/overlay-play.png)

![Setup: model sources, devices and execution routes](https://raw.githubusercontent.com/RazvanManolache/dlss5-phoenix-addon/0.15/docs/features/overlay-setup.png)

These are renders of the actual shared C++ ImGui panel in a CPU fixture, with synthetic status. They are not game captures or performance measurements. More views are in the [README gallery](USER_GUIDE.md#a-quick-look).

## Downloads and installation

| Download | Contents |
| --- | --- |
| `DLSS5-Phoenix-0.15-Windows.zip` | Standard x86/x64 add-ons and rebuilt graphics helper; retained single-target RTX 50 kernels |
| `DLSS5-Phoenix-0.15-RTX-Compatibility-Windows.zip` | Alternative experimental x86/x64 package with retained `sm_75/86/89/120` kernel images; older physical cards remain unvalidated |
| `DLSS5-Phoenix-0.15-Portable-Runtimes.zip` | Optional architecture-matched runtime companions, retained from 0.14 |
| `DLSS5-Phoenix-0.15-Input-Windows.zip` | Separate optional `PhoenixInput.addon32/64`, instructions, licenses and validation receipt |
| `DLSS5-Phoenix-0.15-Source.zip` | Audited source, tests, documentation, screenshots and licenses; no model data or compiled kernels |

1. Close the game. Use a current [ReShade build with full add-on support](https://reshade.me/).
2. Choose **one** main Windows package. Replace the existing `Phoenix.addon32` or `Phoenix.addon64` matching the game's architecture, plus `phoenix_graphics_host.exe` for helper fallback. Do not load standard and compatibility variants together.
3. Keep your existing configuration and separately obtained compatible model. The new layout does not reset your settings.
4. In x86 games, NVIDIA DLSS5 evaluation and NVIDIA comparisons still require Automatic or External x64 helper; unsupported forced routes explain the requirement.
5. Optional portable runtimes are unchanged from 0.14. An intact installation does not need replacement. Profile-specific WOW64 packages are not included.
6. Install the optional Input companion only if you want automation. Copy the matching `PhoenixInput.addon32/64` beside ReShade; see [Phoenix Input](PHOENIX_INPUT.md). It is not required for neural rendering.

The packaged helper and its nested x64 addon are rebuilt too, so helper-based execution receives the same fixes and UI. The installed package requires no Python, compiler or CUDA toolkit.

## Validation boundary

Release validation uses CPU contracts, software D3D12, shared-panel interactions, real DLL load/unload against a minimal ReShade host, mailbox checks, resource extraction and recursive content/hash audits. No new game runs or hardware GPU inference benchmarks are performed for this publication.

Native kernel images and portable runtime payloads are retained from the verified 0.14 packages. Their earlier qualification is not relabeled as a fresh 0.15 result. Input companion validation covers startup, status/session handling and explicit clean unload; broad gameplay/input-API compatibility remains experimental.

No NVIDIA model DLLs, model weights, private configurations or personal captures are included. Archives include file inventories and SHA-256 checksums.

[Validation](VALIDATION_0.15.md) · [Overlay guide](overlay-ui.md) · [Model switching](model-switching.md) · [Execution choices](EXECUTION_MECHANISMS.md) · [Support](SUPPORT.md) · [Credits](UPSTREAM_CREDITS.md)
