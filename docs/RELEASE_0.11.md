# Phoenix DLSS5 0.11

Transport choices now reflect what the current game, architecture, settings and
runtime packages can actually use. The regular same-device native DX12 executor
is x64-only; the separate x86 experimental implementations need matching packages.

- Automatic tries eligible same-process routes, then the external x64 helper.
  Failed attempts are visible and can be retried after correcting the setup.
- Configured native/portable runtime folders take precedence over default locations.
  Discovery verifies required DLL architecture, declared hashes, Windows/driver
  identity and native-driver graph readiness without loading them on the render thread.
- Older experimental x86 packages without model declarations are offered for
  Original only. The driver route also needs its tile-aligned graph and an
  already large-address-aware game executable; Phoenix never patches the game.
- Missing or unsupported choices have an explanation instead of silently
  pretending to be another route. Model DLLs can remain outside the game folder.

**Install:** use a ReShade build with full addon support. Copy the matching
`Phoenix.addon32` or `Phoenix.addon64` and `phoenix_graphics_host.exe` beside the
game's ReShade installation. Supply your compatible model separately.
Optional portable runtimes are a separate download. Profile-specific WOW64
packages are not included in this release.

Same-process first is a user preference, not a speed guarantee. The helper was
faster than the experimental x86 routes in the retained diagnostic fixture
samples. Native neural kernels and optional portable inference libraries retain
the prior payloads; this update makes no new throughput or image-parity claim.

[Validation scope](VALIDATION_0.11.md) · [Execution mechanisms](EXECUTION_MECHANISMS.md)

No NVIDIA model DLL, model weights, private configuration or captures are included.
