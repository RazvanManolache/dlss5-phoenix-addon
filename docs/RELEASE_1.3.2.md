# Phoenix DLSS5 1.3.2

- **Optional in-process async worker.** Supported 64-bit DX11 and DX12 games can keep presenting while Native RTX processes final color. Select **In-process async worker** under **Execution mechanism** for compatible single-sample SDR backbuffers. Use the direct route for engine guides or NVIDIA DLL output.
- **Improved x86 refinement and frame delivery.** Updated native routes reuse pixel buffers and keep intermediate refinement passes on the GPU. DX9 pacing gives the game more room to present when refinement work is queued.
- **Safer Windows routing.** Phoenix checks the game’s render adapter for same-device processing and switches memory-starved 32-bit games to the external x64 helper when that route is compatible.

Update the addon and its optional WOW64 runtime together. DLSS5 Everything can install and verify the matching files. See the [installation guide](INSTALLATION.md) and [validation details](VALIDATION_1.3.2.md).
