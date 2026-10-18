# Phoenix add-on 0.82

This update adds experimental Phoenix appearance controls and fixes native
1024x768 first-frame failure and inconsistent 4K reset output.

## Changes

- Phoenix appearance enable/disable, intensity from 0 to 2, and Default,
  Natural and Cinematic styles in the shared panel. Phoenix settings are
  persisted separately from NVIDIA overrides.
- Experimental per-surface control maps, protected UI composition and
  bidirectional distortion inputs through versioned library/feeder interfaces.
  These require an engine producer; the add-on cannot identify arbitrary
  objects from final color alone. The final-color portable route hides options
  whose input textures it cannot supply.
- Temporal-gain and sampling controls, bounded optional-plane transport/replay,
  model/resize transaction checks and producer connection status.
- Corrected graph selection and embedding: complete 32-token storage and
  persistent spatial padding now accompany the native kernels. Old graph
  layouts fail before execution, and packaging rejects stale graph metadata.
- D3D12 errors identify the HRESULT and failing operation.

[Features and pipeline](NEURAL_CONTROLS.md) and
[the native correctness investigation](NATIVE_CORRECTNESS_0.82.md) explain the
contracts, diagnosis and verification boundaries.

## Downloads

- `DLSS5-Phoenix-0.82-Windows.zip`: x64/x86 ReShade add-ons, graphics helper,
  embedded model-free runtime/kernels and notices.
- `DLSS5-Phoenix-0.82-Portable-Runtimes.zip`: matching standalone Windows
  runtimes, import libraries, CMake packages and public headers for integrations
  and explicit experiments. The x64 `bin/native` folder holds the matching
  model-free graph/kernel bundle.
- `DLSS5-Phoenix-0.82-Source.zip`: audited source matching the release commit.
- `VALIDATION.json` and `SHA256SUMS.txt`: scope, inventories and checksums.

Use a current ReShade build with full add-on support. Copy only the matching
`Phoenix.addon32` or `Phoenix.addon64` and the graphics helper beside ReShade,
keep the supplied notices, and select your separately obtained compatible model
in the Phoenix panel. Start with automatic transport. No NVIDIA model DLL,
neural weights, installed-game settings or system/driver DLLs are supplied.

The 0.81 experimental WOW64 profile is not revalidated or republished here.
Its profile-specific limitations remain; use the x64 helper when needed.

## Evidence and limits

Native correctness checks cover immediate neighbors of 1024x768 and 4K, plus
common sizes through 8K. Three explicit resets per session, separated by eight
accepted continuation frames, produce exact repeated output in the fixed
fixtures. Earlier performance tables retain their recorded pre-fix runtime
identities and are not fresh 0.82 speed measurements.

The new artistic and optional-input contracts remain experimental. Vendor
output parity, arbitrary HDR UI, real moving engine objects and broad device
acceptance are not established by these fixtures. The portable optional-effects
path has substantial CPU processing costs. No retail game was used for this
release validation. See [validation details](VALIDATION_0.82.md).
