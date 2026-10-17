# Add-on 0.81

This is a Windows ReShade add-on release, not a renumbering of the standalone
SDK's `0.8.1-beta` or the Android preview. Project code remains Apache-2.0.

## Highlights

- x86 and x64 add-ons with renderer-first controls and Advanced transport below.
- Retained automatic/native D3D12 and external x64-helper execution.
- Experimental in-process portable execution with architecture-matched runtimes.
- Opt-in x64 Phoenix/NVIDIA library execution inside an x86 process.
- Opt-in x86 Phoenix graph host calling a native x64 driver executor in-process.
- Live strengths, same-evaluation comparisons, FP16 intermediate refinement,
  explicit fallback reasons and separate game/output/execution diagnostics.
- D3D9 multisample capture resolve and split Vulkan graphics/present queues.
- Worker retirement/TLS recreation and x86 proxy dependency-loading fixes.
- Full-width portable GPU handles and undecorated Windows x86 C API exports.
- Preserved Android approximate Vulkan/tiled-kernel source changes.
- [Execution benchmarks](benchmarks/0.81/README.md) with separate timing scopes.

## Downloads and Installation

1. Install a current **ReShade build with full add-on support** for the game API
   from [reshade.me](https://reshade.me/). Respect anti-cheat and multiplayer rules.
2. Extract `DLSS5-Phoenix-0.81-Windows.zip`. Copy **only the matching**
   `Phoenix.addon32` or `Phoenix.addon64` next to the game's ReShade DLL.
3. Keep `phoenix_graphics_host.exe`, `licenses/` and the supplied notices beside
   the add-on. The embedded helper payload is retained; no replacement ReShade
   DLL for the game is included.
4. Open ReShade's Phoenix panel and select your separately obtained compatible
   model. No NVIDIA model DLL, model weights or Windows/driver DLL is supplied.
5. Choose the renderer. Leave Advanced transport on **Automatic** initially.
   Incompatible paths show their reason and follow the configured fallback.
6. For portable experiments, extract the separate
   `DLSS5-Phoenix-0.81-Portable-Runtimes.zip` beside the add-on, preserving
   `PhoenixRuntime/win-x86` and `PhoenixRuntime/win-x64`. These retained runtime
   builds have their own validation history; unvalidated devices require explicit
   opt-in. CPU execution is a reference path, not a real-time performance promise.
7. For native x86 experiments, extract
   `DLSS5-Phoenix-0.81-Experimental-WOW64.zip` there as well. Select **x64 library
   inside x86** or **x86 Phoenix + native driver**. The included bootstrap matches
   one verified Windows/driver profile only. On mismatch use the helper; never
   disable the checks or copy system DLLs from another computer.

```text
game/
  <game executable>
  <ReShade proxy DLL>
  Phoenix.addon32                 # OR Phoenix.addon64
  phoenix_graphics_host.exe
  licenses/
  PhoenixRuntime/                 # optional extra packages
    win-x86/
    win-x64/
    wow64/
```

The x86 graph-host path is Phoenix-only, full eight-block network once, and
requires an already large-address-aware executable. It never patches the game
EXE. NVIDIA views on this path use the retained helper. The x64-library-inside-x86
path can evaluate both renderers. Both require the Native RTX executor.

## Evidence and Limits

Both add-on architectures compile and pass focused CPU contracts. Packaged x86
client inference, worker switching, live strengths and comparisons were tested
with the locally verified runtime profile. **Real game presentation, resize,
device-loss safety and broad Windows/driver compatibility are not established
for the new experimental paths.** Default automatic routing is unchanged.

Native kernels currently target **sm_120**, measured on an RTX 5080. In-process
x86 transports still stage pixels through CPU memory, create their own D3D12
device and are not crash-isolated. Their resident-input benchmark is not an
end-to-end add-on latency benchmark. See [mechanism documentation](EXECUTION_MECHANISMS.md).

Release archives exclude user configs, local profiles/logs, captures, model
weights and NVIDIA model DLLs. The optional WOW64 package includes the unmodified
LGPL-3.0 WOW64Ext dynamic dependency, its corresponding source and license.
See [third-party notices](../THIRD_PARTY.md). Public benchmark summaries are
curated measurements, not raw local diagnostic receipts.

[Release validation details](VALIDATION_0.81.md) distinguish fresh CPU/build
checks from retained inference and performance evidence.
