# Phoenix DLSS5 0.10

Software-rendered games can use a hardware helper without trying to run the
native RTX executor on a software adapter. Automatic selection recognizes WARP
devices; an explicitly requested unsupported in-process route reports its limitation.
Software D3D11 capture uses readback rather than incompatible shared textures.

- Matching x86/x64 add-ons and the x64 graphics helper are included.
- Optimized neural kernels and optional portable runners retain their 0.9 payloads.
- The NVIDIA model DLL remains separately supplied; no models or vendor SDK binaries are included.
- For installation, copy `Phoenix.addon32` or `Phoenix.addon64` alongside an
  add-on-capable ReShade installation. Standalone portable runners remain optional.
- DLSS5 Everything 3.1 adds an opt-in ReShade software-device hook for games that need it.

[Validation scope](VALIDATION_0.10.md) and [previous installation details](RELEASE_0.9.md).
