# Compatibility and execution choices

[Project overview](../README.md) · [Documentation index](README.md)

Architecture and renderer choices have different requirements. The [support matrix](SUPPORT.md) records which devices/routes were actually validated.

## 32-bit Games: x86 Compatibility

**Yes, Phoenix DLSS5 also has a 32-bit game add-on.** Use `Phoenix.addon32` for an
x86 game and `Phoenix.addon64` for an x64 game. Both are in the
**[Windows download](https://github.com/RazvanManolache/dlss5-phoenix-addon/releases/tag/1.0-beta)**.
This is support for 32-bit games on **64-bit Windows**, not 32-bit Windows or
older GPUs without the required rendering capabilities.

**Start with Automatic.** For an x86 game, the retained compatibility route
captures the game image, sends it to a separate x64 helper running Phoenix DLSS5 or
NVIDIA DLSS5, and displays the processed result back in the game. The game stays
32-bit; the neural renderer runs in the helper. No game EXE patch is needed.
The extra image transfers add overhead, even when model inference is fast.

Two **optional experiments** keep native processing inside
the x86 game's process instead of using that separate helper:

| x86 execution choice | What happens | Renderers |
| --- | --- | --- |
| External x64 helper | The game sends frames to a separate 64-bit processing program | Phoenix DLSS5 and NVIDIA DLSS5 |
| x64 library inside x86 (experimental) | A native 64-bit worker runs Phoenix inside the 32-bit game's process | Phoenix DLSS5 only; NVIDIA DLL/comparison uses Automatic or the x64 helper |
| x86 Phoenix DLSS5 + native driver (experimental) | A 32-bit Phoenix DLSS5 host prepares the network work; a native 64-bit executor submits it to the GPU | Phoenix DLSS5 only; NVIDIA DLSS5 views use the helper |

The experiments passed packaged-client inference, controls and switching tests
on the verified Windows/driver profile. **Game presentation, resize and
device-loss behavior for these new paths are not yet validated.** They require
the separate experimental runtime package and a matching Windows/driver profile;
Automatic considers them only when their prerequisites pass. Leave helper fallback on. The x86 Phoenix DLSS5
host also requires an already large-address-aware game EXE and does not patch it.
Both native experiments still stage CPU pixels: in-process is not zero-copy,
crash isolation or a guaranteed performance improvement.

[Installation, configuration and exact limits](EXECUTION_MECHANISMS.md)

## Execution Choices

The release includes **both x86 and x64 ReShade add-ons**, the retained x64
helper, and optional experimental runtimes. Models and NVIDIA DLSS5 DLLs stay external.

- **Choose where rendering runs.** Automatic, native in-process D3D12, external
  x64 helper, or experimental portable/native x86 execution. Automatic selection
  remains conservative; the helper stays available.
- **Renderer first, transport second.** Choose Phoenix DLSS5, NVIDIA DLSS5 or comparison
  first; **Advanced transport** is underneath, with descriptive names instead of
  experiment numbers. Unsupported combinations explain their helper fallback
  without discarding your saved choice.
- **Change the look live.** Tone, structure, skin and mask strengths update
  without restarting the model on every slider movement. Experimental native x86
  paths support refinement passes with FP16 intermediate inputs. The x64-library
  path supports block repetitions; the x86-host path currently uses all eight
  blocks once. Unsupported controls are disabled or explicitly rejected.
- **Compare the same frame.** Split and difference views use the input snapshot
  belonging to that evaluation, rather than a newer game frame. Diagnostics
  distinguish game presentations, completed model outputs, execution location,
  inference PID, helper PID and processing round-trip time.
- **Broader capture compatibility.** D3D9 MSAA resolve, separate Vulkan graphics
  and presentation queues, and capture integration for D3D9/10/11/12, OpenGL and
  Vulkan. Portable x86 fixes preserve full-width GPU handles and C API exports.
- **Safer worker lifecycle.** Model loading and inference stay off the game
  presentation thread, retirement does not join a worker there, native TLS is
  initialized for replacement workers, and x64 dependencies cannot accidentally
  resolve to a game's x86 DXGI proxy.
- **Android work preserved.** The standalone SDK source retains the explicit
  approximate mobile Vulkan arithmetic and tiled fast-kernel options from the
  separate [Android preview](VULKAN_FAST_MOBILE.md). These are not native RTX
  performance claims or new Android binaries in this add-on release.

[Execution configuration and limits](EXECUTION_MECHANISMS.md) |
[Historical execution benchmark data and methodology](benchmarks/0.81/README.md)

<details>
<summary>RTX generations, compatibility DLLs and validation limits</summary>

The model picker also accepts the **OpenNR FP16 compatibility build of
310.9.16.0**, downloaded centrally by DLSS Everything from
[NODIX DLSS-5-MANAGER 1.3.7](https://github.com/NODIX-TECH/DLSS-5-MANAGER/releases/tag/1.3.7).
Its Original and Lightweight model resources match the NVIDIA reference byte
for byte. NVIDIA's original DLLs and shared native model exports remain supported.
The compatibility DLL matches [OpenNR's published identity](https://github.com/clshortfuse/openNR/blob/main/profiles/compat-fp16-310.9.16.json)
(`cdfa6c1e…f3ed`); it is community-modified runtime code, not another trained model.

**RTX 20 / 30 / 40 / 50 series support:** the optional **RTX Compatibility** addon supports both
**Original and Lightweight** through Phoenix's Native RTX backend, and through
the NVIDIA DLSS5 backend when using the **patched OpenNR compatibility DLL** above.

| GPU generation | Phoenix Native RTX | NVIDIA DLSS5 with patched compatibility DLL |
| --- | --- | --- |
| RTX 20 | Experimental `sm_75` compatibility kernels | Experimental compatibility path |
| RTX 30 | Experimental `sm_86` compatibility kernels | Experimental compatibility path |
| RTX 40 | Experimental `sm_89` kernels | Experimental compatibility path |
| RTX 50 | Existing `sm_120` kernels; tested on RTX 5080 | Supported device images; patched DLL execution untested here |

For Phoenix, the NVIDIA driver automatically selects the matching kernels for
the active processing GPU; no older-card switch is required. For NVIDIA DLSS5,
select the patched compatibility DLL as the runtime to use its compatibility
code. NVIDIA's original DLLs remain supported with their own hardware requirements.
Older RTX 20/30/40 inference, visual quality and performance still need testing on
those physical cards. The current tuned RTX 50 images are rebuilt and checked
on RTX 5080; see [native architecture support](NATIVE_ARCHITECTURES.md).
OpenNR's host patches cover D3D11/D3D12; its
[support boundary](https://github.com/clshortfuse/openNR#support-boundary)
does not claim CUDA/Vulkan backend compatibility. These additions require the
RTX Compatibility package. The standard Windows package retains the
single-target RTX 50 native payload. Do not install both add-ons together.

A local **experimental x86/x64 addon** was built with all four targets and the
selectable instruction styles. It passed 25 CPU suites, four software graphics
suites, DLL lifetime checks and embedded-payload audits. Its pinned kernel bundle
passed both-model reset/history checks from small frames through 8K on RTX 5080.
[Build and validation details](NATIVE_ARCHITECTURES.md).
These isolated development checks do not measure every packaged route or game.
Older RTX cards still need physical-device tests; [package validation](VALIDATION_0.19.md)
separates the fresh package checks from earlier measurements.

</details>
