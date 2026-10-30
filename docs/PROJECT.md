# Why Phoenix DLSS5 exists

[Project overview](../README.md) · [Documentation index](README.md)

Project motivation, contributors, upstream work and licensing.

## Why I'm making this

I dislike how much of the software we add to our games is closed source. We
rush for the best experience, often giving code access to our systems without
being able to inspect what it does—or what that means for our privacy. I'd
rather let you look under the hood. Better pixels shouldn't require blind trust.

- **Credit where it's due:** NVIDIA built something impressive. Their model,
  Tensor Cores and native execution set a very high bar; optimized RTX is easily
  this project's strongest path today. The goal is to explore that technology
  across more runners and expose useful controls.
- **Why bother?** I know DLSS 5 hasn't exactly won a popularity contest. I'm
  optimistic about [NVIDIA's approach to neural rendering](https://www.nvidia.com/en-us/geforce/news/dlss-5-3d-guided-neural-rendering/):
  my bet is that convincing real-time realism needs more than endlessly larger
  meshes, textures and rendering budgets. Asking every ordinary studio to build
  movie-quality assets and lighting consistently is a formidable bill.
  Artist-directed neural rendering looks like a promising way forward.
- **More artistic choice would be lovely.** I'd like people to train compatible
  models for their own looks: cartoons, painterly worlds, or something less
  interested in counting pores. Phoenix DLSS5 already supports selecting a compatible
  model from disk. There is **no trainer or alternative trained model included**;
  replacement weights must fit the supported architecture and layout.
- **A very personal hardware opinion:** after the GPU price hikes, the RTX 5080
  is the only card I'd currently consider reasonably priced for the high-end
  gaming I want. “Reasonable” is doing heroic work in that sentence.

## Credits and upstream work

Phoenix builds on other people's work. The following distinguishes adapted
source, ports of published designs, and references we studied:

| Upstream | What Phoenix uses or learned |
| --- | --- |
| [mauri870/dlssnr-pytorch](https://github.com/mauri870/dlssnr-pytorch) | Adapted tensor-layout declarations retained after the native C++ migration; MIT attribution remains. |
| [Uzbekunknown/dlss-nr-on-intel](https://github.com/Uzbekunknown/dlss-nr-on-intel) | Imported shader/runtime source subset, with recorded modifications; Apache-2.0. |
| [mochizuki0323/DLSSNR-AMD](https://github.com/mochizuki0323/DLSSNR-AMD) | Imported cooperative-matrix and activation shader interfaces; MIT. |
| [kibblerz/DLSS5-Reshade-AIO](https://github.com/kibblerz/DLSS5-Reshade-AIO) | Directly adapted native NVIDIA Optical Flow provider, integrated into Phoenix's queue/history handling; Apache-2.0, with NVIDIA MIT interface headers. |
| [NVIDIA Image Scaling](https://github.com/NVIDIAGameWorks/NVIDIAImageScaling) | Retained SDK scaling/sharpening shaders and configuration code; MIT. The D3D12 adapter is Phoenix code. |
| [Benjamin-Wegener/dlss5-distill](https://github.com/Benjamin-Wegener/dlss5-distill) | Published tiny 2x architecture and optional trained checkpoint, ported to native D3D12; Apache-2.0. The C++/HLSL implementation is new. |
| [clshortfuse/openNR](https://github.com/clshortfuse/openNR) | Older-RTX compatibility approach, including FP8-to-paired-FP16 decomposition and architecture lowering; retained MIT attribution. |
| [Yukikaze1945/Lunarc-DLSS5](https://github.com/Yukikaze1945/Lunarc-DLSS5), [Yukikaze20170315/dlss5-vulkan-fg-relay](https://github.com/Yukikaze20170315/dlss5-vulkan-fg-relay) | Studied native media/scaling, multiscale processing, pipeline ordering and recovery designs. These are references, not imported implementations. Phoenix's independently implemented native chain/fusion has small-fixture checks; VSR remains deferred. |

The companion video player adapts the native player/media/cache/comparison
foundation from [2600th/dlss5-video-player](https://github.com/2600th/dlss5-video-player)
(MIT). Everything's driver-profile adapter is adapted from
[ApolloF/DLSS5-Updater](https://github.com/ApolloF/DLSS5-Updater) (MIT).
Those adaptations live in the companion projects, not in Phoenix's addon source.

See the [upstream credit ledger](UPSTREAM_CREDITS.md) for source locations,
recorded revisions, additional research references and external model sources.
[THIRD_PARTY.md](../THIRD_PARTY.md) links retained licenses and notices, including
ReShade, ImGui, MinHook, JSON and vendor interface dependencies. Credits identify
contributions; they do not imply upstream endorsement or completed integration.
NVIDIA created DLSS and the supplied NVIDIA models, which retain their own terms.

## Help wanted

- Sorry, discrete Radeon and Intel GPU owners: I don't have those cards to test.
  AMD HIP/Vulkan passed on the integrated Radeon, and SYCL was tested on the
  Ryzen CPU through Intel's CPU runtime. **Neither establishes discrete AMD or
  Intel GPU compatibility or performance.** The portable paths are intended to
  work on devices with the required capabilities; hardware testing is welcome.
- Contributions are welcome: faster portable kernels, AMD/Intel hardware
  results, dual-GPU testing, compatible model experiments, bug reports and docs.
  Please include hardware, driver, resolution, settings and reproducible results.
- This is an independent experimental project, unaffiliated with NVIDIA.
  NVIDIA deserves the credit for DLSS and its models.

## License

Original project code is licensed under [Apache 2.0](../LICENSE).
Third-party components and adaptations retain their upstream terms; see
[THIRD_PARTY.md](../THIRD_PARTY.md) and [NOTICE](../NOTICE).
The project license does not grant rights to separately supplied NVIDIA DLSS5
model DLLs, model data or vendor SDKs/runtimes.
