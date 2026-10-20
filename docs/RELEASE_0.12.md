# Phoenix DLSS5 0.12

More RTX generations can join the experiment. Both Original and Lightweight
now have architecture-specific native kernels for RTX 20, 30, 40 and 50 cards.
The older-card paths are experimental, not hardware-qualified compatibility claims.

## Choose Your Package

| Download | Use |
| --- | --- |
| `DLSS5-Phoenix-0.12-Windows.zip` | Standard x86/x64 add-ons and helper; validated RTX 50 native kernel payload |
| `DLSS5-Phoenix-0.12-RTX-Compatibility-Windows.zip` | Experimental x86/x64 add-ons and helper; compiled `sm_75/86/89/120` images for RTX 20/30/40/50 |
| `DLSS5-Phoenix-0.12-Portable-Runtimes.zip` | Optional architecture-matched runtime companions, unchanged from the preceding verified payloads |
| `DLSS5-Phoenix-0.12-Source.zip` | Audited source, tests, documentation and licenses; no model weights or compiled kernels |

Use a current ReShade build with full add-on support. Close the game, choose
one Windows package, and copy only the matching `Phoenix.addon32` or
`Phoenix.addon64` plus `phoenix_graphics_host.exe` beside ReShade. Replace the
previous Phoenix add-on; do not load standard and compatibility copies together.
Keep your model files and configuration. The installed runtime needs no Python,
compiler or CUDA toolkit. Profile-specific WOW64 packages remain separate.

## RTX Compatibility

- RTX 20 uses FP8 storage decoded into FP16 Tensor Core steps (`sm_75`).
- RTX 30 uses paired FP16 matrix steps (`sm_86`).
- RTX 40 uses native FP8 matrix instructions (`sm_89`).
- RTX 50 retains the exact 41 validated `sm_120` images.
- The NVIDIA driver selects the processing device's image when loading each
  module. No per-frame architecture switch or model replacement is added.
- Both add-on architectures retain the experimental identity and overlay warning.

Offline checks cover all 164 images. Physical RTX 20/30/40 inference, driver
behavior, image quality and performance have not been validated here. Unsupported
devices still fail through the normal route handling; a compilation is not proof
that a particular card or game works. The regular stable embedding gate remains strict.

The model picker additionally accepts the pinned OpenNR FP16 compatibility DLL
distributed by [NODIX Manager 1.3.7](https://github.com/NODIX-TECH/DLSS-5-MANAGER/releases/tag/1.3.7).
Its weights match the existing models; its executable runtime code is community
modified. NVIDIA DLSS5 execution through that DLL is separate from Phoenix's
native compatibility kernels and has not been qualified on older cards here.
OpenNR documents D3D11/D3D12 host patches, not general CUDA/Vulkan compatibility.
Original NVIDIA editions and compatible native model folders remain supported.

The x86 OpenGL route selector also excludes known-crashing mixed-bitness native
mechanisms; the helper and eligible portable choices remain available.

[Validation](VALIDATION_0.12.md) · [Architecture details and attribution](NATIVE_ARCHITECTURES.md)
· [Execution choices](EXECUTION_MECHANISMS.md) · [Support matrix](SUPPORT.md)

No NVIDIA model DLLs, weights, logs, private configuration or game captures are
included. SHA-256 checksums accompany every downloadable package.
