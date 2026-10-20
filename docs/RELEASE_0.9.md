# Phoenix DLSS5 0.9

Two model sizes, more control, less housekeeping. Native RTX is still the main
event; the portability side quest comes with its own honest limits.

## What's New

- Original and Lightweight FP8 from compatible 310.9.16 model sources, with
  explicit model selection. Original sources from 310.8 remain supported.
- Compatible external model folders and a shared native model library. Select
  weights centrally instead of copying a large NVIDIA DLL into every game.
  New network architectures still require an implementation, not just a file.
- Model-aware controls, block scheduling, repeated passes and adaptive budgets
  on supported routes. Unsupported controls remain unavailable.
- Cached schedules, lazy appearance buffers, lower startup allocation peaks,
  safer Vulkan defaults and corrected history/boundary handling.
- Structured native kernels, documented contracts and additional regression
  tests, without replacing the measured production arithmetic.
- Refreshed Original/Lightweight performance, RAM/VRAM, source-parity and
  released-versus-current reports. The measured development binaries retain
  their identities; rebuilding does not retroactively change those results.

## Performance

At 1080p on the tested RTX 5080, native Phoenix DLSS5 inference is on par with
NVIDIA DLSS5 for both models. The 20 October fixture measured 3.874 ms GPU time
for Original and 2.282 ms for Lightweight within Phoenix DLSS5, versus 3.825 ms
and 2.223 ms for their NVIDIA DLSS5 counterparts. Small margins vary across
runs. Original at 4K has a larger difference; this is not a universal speed or
game-FPS claim.

Both implementations matched the corresponding NVIDIA preset's reset/history
fixture outputs from 320x180 through 8K. Shared model exports matched their DLL
sources in the tested routes. Neither result establishes moving-scene quality.
See the [full measurements and memory report](benchmarks/current/README.md).

## Installation

1. Install [ReShade with full add-on support](https://reshade.me/).
2. Extract `DLSS5-Phoenix-0.9-Windows.zip`. Copy only the add-on matching the
   game's process architecture beside its ReShade DLL, together with
   `phoenix_graphics_host.exe`. Keep the supplied notices.
3. For x64 NVIDIA DLSS5 comparison, name the active add-on
   `Phoenix_nvngx.dll.addon64`. Use only one active copy. NVIDIA checks the
   calling module's name; renaming the model DLL is not sufficient.
4. Open ReShade's Phoenix panel and select a separately obtained compatible
   model DLL or authenticated model folder. Select Original or Lightweight.
   A shared model library is optional; arbitrary architectures are not supported.
5. Start with automatic transport. Native x64 D3D12 can run in process;
   compatibility routes retain the x64 graphics helper. An x86 process cannot
   load ordinary x64 libraries through the Windows loader.
6. Install `DLSS5-Phoenix-0.9-Portable-Runtimes.zip` only when using a standalone
   library or explicitly selected portable route. Keep its `PhoenixRuntime`
   directory beside the add-on. Match the provider architecture to the process.

Example x64 installation:

```text
Game/
  Game.exe
  dxgi.dll                       # ReShade, obtained separately
  Phoenix_nvngx.dll.addon64       # one active Phoenix add-on
  phoenix_graphics_host.exe
  PhoenixRuntime/                # optional standalone/portable package
    win-x64/
    win-x86/
```

Models may remain in a separately configured central directory. NVIDIA model
DLLs, weights, system DLLs, logs and private game configurations are not supplied.
The [execution guide](EXECUTION_MECHANISMS.md) explains x86 transport choices.
Profile-specific experimental WOW64 components are not included in this release;
do not assume a package built against another Windows/driver profile will work.
Use automatic/helper transport when the required optional mechanism is absent.

## Downloads And Evidence

- `DLSS5-Phoenix-0.9-Windows.zip`: matching x86/x64 add-ons and graphics helper,
  embedded model-free runtime/kernels, documentation and dependency notices.
- `DLSS5-Phoenix-0.9-Portable-Runtimes.zip`: matching optional Windows runtime
  components and notices; no models or operating-system/driver binaries.
- `DLSS5-Phoenix-0.9-Source.zip`: audited source at the single release commit.
- `VALIDATION.json`: exact package checks and their limits.
- `SHA256SUMS.txt`: asset integrity checksums.

The [support matrix](SUPPORT.md), [model guide](MULTI_MODEL.md),
[shared-library guide](SHARED_MODELS.md) and [controls guide](PORTABLE_MODEL_CONTROLS.md)
distinguish native, portable and experimental capabilities. No retail game is
launched or modified during release-package validation. Read `VALIDATION.json`
for the checks performed on the distributed binaries, separately from historical
benchmark and inference receipts. [Validation scope and known limits](VALIDATION_0.9.md).
