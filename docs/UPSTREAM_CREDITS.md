# Upstream credits and research references

This ledger covers Phoenix, DLSS Everything and the companion Phoenix Video
Player. It distinguishes actual source reuse, a native port of a published
design, external model/runtime sources, and projects studied during development.
Source similarity alone is not treated as proof of copying or of a particular
reverse-engineering method. Existing file headers and license notices remain
authoritative for their components.

## Source reused or designs ported in Phoenix

| Project | Contribution and local scope | Origin record |
| --- | --- | --- |
| [mauri870/dlssnr-pytorch](https://github.com/mauri870/dlssnr-pytorch) | Adapted tensor-layout declarations. The earlier imported Python runtime was removed during the native migration; its retained declarations remain credited. | MIT; [provenance and reference revision](../third_party/portable_nr/PROVENANCE.md), [notice](../third_party/portable_nr/NOTICE). |
| [Uzbekunknown/dlss-nr-on-intel](https://github.com/Uzbekunknown/dlss-nr-on-intel) | Selected shader and runtime/reference source subset in `third_party/intel_nr/`, with documented adaptations. | Apache-2.0; pinned `0b43a76a3b8e34ad88859dedfd2751b5028be982`; [file-level provenance](../third_party/intel_nr/PROVENANCE.md). |
| [mochizuki0323/DLSSNR-AMD](https://github.com/mochizuki0323/DLSSNR-AMD) | Cooperative-matrix and coherent-activation GLSL interface headers in `third_party/amd_nr/`. | MIT; pinned `9e4574e1812d7d0632d6db02721a20c9617be434`; [file-level provenance](../third_party/amd_nr/PROVENANCE.md). |
| [kibblerz/DLSS5-Reshade-AIO](https://github.com/kibblerz/DLSS5-Reshade-AIO) | Directly adapted optical-flow provider in `src/pipeline/nvof/`; shared adapter in `src/pipeline/d3d12_motion.h`. Adaptations include inline implementation, system-driver loading and Phoenix queue/history ownership. | Apache-2.0 provider; NVIDIA interface headers retain MIT notices. [License and adaptation notice](../third_party/aio-motion/NOTICE). Inspected source revision: `09301f5528e619e8b9ec17c257d167e2985f53b0`. |
| [NVIDIA Image Scaling](https://github.com/NVIDIAGameWorks/NVIDIAImageScaling) | Retained SDK shaders/configuration in `third_party/nis/`; embedded shader source comes from those SDK files. `src/pipeline/nis.h` is the Phoenix D3D12 adapter. | MIT; inspected SDK revision `35e13ba316c98eeecf16f37eae70ce88019911f6`; [notice](../third_party/nis/NOTICE). |
| [Benjamin-Wegener/dlss5-distill](https://github.com/Benjamin-Wegener/dlss5-distill) | Published Phase4Generator deploy architecture and optional trained coefficients. `src/pipeline/tiny.h` and `tiny_shader.h` are a new native C++/HLSL implementation; conversion imports the pinned checkpoint without a deployed Python runtime. | Apache-2.0, Benjamin Wegener; pinned `9e3bc2f9e755f0e281b72e439f9452be0e8b7f31`; [notice](../third_party/tiny-distill/NOTICE). This is a separate tiny upscaler, not NVIDIA NR. |
| [clshortfuse/openNR](https://github.com/clshortfuse/openNR) | Compatibility approach studied and applied to architecture fallbacks, especially FP8-to-paired-FP16 matrix decomposition. Phoenix uses its own compile-time CUDA helpers and existing synchronous-copy path. | Retained [MIT license](../third_party/opennr/LICENSE); [scope in the dependency inventory](../THIRD_PARTY.md), [implementation description](NATIVE_ARCHITECTURES.md). The OpenNR DLL builder is not a runtime dependency. |
| [DLSS Everything / Compatibility Installer](https://github.com/RazvanManolache/dlss5-everything) | Compatibility host and transport source adapted from the local `DLSS5-Feeder-clean-20260901-v2` snapshot. | MIT; [notice](../third_party/compatibility_bridge/NOTICE) and import manifest record the snapshot and maintained paths. The snapshot label is a retained origin declaration, not an authenticated upstream commit. |

ReShade, Dear ImGui, MinHook, nlohmann JSON, WOW64Ext and vendor interface
dependencies have separate entries and retained license texts in
[THIRD_PARTY.md](../THIRD_PARTY.md). This ledger supplements those notices.

## Direct adaptations in the companion projects

| Project | Contribution | Recorded source and license |
| --- | --- | --- |
| [2600th/dlss5-video-player](https://github.com/2600th/dlss5-video-player) | Native Windows player, decoder/encoder orchestration, media/cache foundation, comparison renderer, subtitles and UI imported into the separate `dlss5_video_player` project, then adapted for direct Phoenix execution. This is substantial source reuse, not merely inspiration. | MIT, DLSS Video Player contributors; pinned `8dfded2bc16477eb985d4d21575832bde07b3df0`. The player retains `third_party/video-player/LICENSE`, `ORIGIN.json` and upstream third-party notices. |
| [ApolloF/DLSS5-Updater](https://github.com/ApolloF/DLSS5-Updater) | NVAPI DRS interop adapted into Everything's `NvDrs.cs`, with Phoenix's own settings journal, restoration and UI layered around it. | MIT; Everything retains `licenses/ApolloF-NvDrs/`. Inspected source revision: `43663f6bce64f050d89d83e1f235e41437ec48c8`. |
| [rakanki911/DLSS5-Swapper](https://github.com/rakanki911/DLSS5-Swapper), [Kizzuwatnaa/DLSS5-Autopilot](https://github.com/Kizzuwatnaa/DLSS5-Autopilot) | Everything's common-action translations and emulator metadata are adapted from Swapper; its emulator table in turn credits Autopilot. Workflow ideas were also studied. | MIT; emulator source pin `9fb0b7c3563d350aa46eaa01d3e5ca4cbab4a7cb`. Everything's source headers and `THIRD_PARTY_NOTICES.md` retain the upstream copyrights and terms. |

Everything also implements the central downloader for the optional dlss5-distill
checkpoint. The actual upscaler executes in Phoenix; this downloader does not
turn NVIDIA NR into the tiny model.

## External model and runtime sources

| Source | How it is used |
| --- | --- |
| [NVIDIA/DLSS](https://github.com/NVIDIA/DLSS), [NVIDIA-RTX/Streamline](https://github.com/NVIDIA-RTX/Streamline) | Official DLSS technology, SDK/interface documentation and SR/RR/FG components. NVIDIA created the NVIDIA networks and model data. These SDK repositories are not described as a source of the NR DLL. |
| [bdsqlsz/DLSSNR3.19.16.0](https://huggingface.co/bdsqlsz/DLSSNR3.19.16.0) | Optional central native exports for the existing Original and Lightweight models. Everything authenticates the selected package; the packing/storage adapter is local code. Package revision: `c305bfc0db1bb94bcc8e25b074904b5158d7cbd8`. These are external NVIDIA-derived model exports, not new Phoenix-trained networks. |
| [NODIX-TECH/DLSS-5-MANAGER](https://github.com/NODIX-TECH/DLSS-5-MANAGER), [OpenNR](https://github.com/clshortfuse/openNR) | NODIX's release distributes the inspected OpenNR FP16 compatibility edition of `nvngx_dlssnr.dll`. Everything downloads the selected runtime member separately; the manager application is not incorporated into Phoenix. Model and modified-runtime terms remain separate from the source-project licenses. |

## References studied; no implementation imported for these features

These credits describe research and design influences, not dependency adoption
or feature-completion claims. In particular, VSR and OptiScaler integration are
deferred. Phoenix's independent processing-chain/fusion implementation has the
route and validation limits recorded in [the pipeline guide](PROCESSING_CHAIN.md).

| Reference | What we studied |
| --- | --- |
| [Yukikaze1945/Lunarc-DLSS5](https://github.com/Yukikaze1945/Lunarc-DLSS5) | Native media processing, VSR bridge requirements, multi-resolution/fusion options and user-facing processing controls. Its VSR bridge and fusion implementation have not been copied into Phoenix. |
| [Yukikaze20170315/dlss5-vulkan-fg-relay](https://github.com/Yukikaze20170315/dlss5-vulkan-fg-relay) | Native NR/FG ordering, routing, timing and recovery behavior. |
| [perseval-BLR/NeuralScreen](https://github.com/perseval-BLR/NeuralScreen) | Screen-capture feature/workflow reference for Everything's Screen Studio. Its PolyForm Strict implementation is not imported. |
| [aloshdenny/open-dlss](https://github.com/aloshdenny/open-dlss), [maanHimself/OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) | Kernel scheduling/fusion, graph structure and matched-output/performance comparisons. No new kernel import from this study is claimed. |
| [RedDukeDev/dlss5-nr-amd-custom](https://github.com/RedDukeDev/dlss5-nr-amd-custom), [sekkit/dlss5-nr-amd-custom](https://github.com/sekkit/dlss5-nr-amd-custom) | Asynchronous result freshness, bounded queues, separate processing/presentation telemetry and AMD runtime organization. |
| [vladmandic/sdnext](https://github.com/vladmandic/sdnext), [SD-Next/DLSS](https://huggingface.co/SD-Next/DLSS) | Readable graph/numerics, bounded temporaries, padding sensitivity and independent output comparisons. |
| [taowen/dlss5-onnx](https://huggingface.co/taowen/dlss5-onnx), [inarikami/dlss5-nr-reverse-engineering](https://huggingface.co/inarikami/dlss5-nr-reverse-engineering) | ONNX quantization, tensor unpacking/operator references and numerical comparisons. Fixed-resolution ONNX variants are not integrated. |
| [alpindale/open-dlss5-nr](https://huggingface.co/alpindale/open-dlss5-nr), [sekkit/open-dlss5-nr](https://huggingface.co/sekkit/open-dlss5-nr) | Export/implementation comparisons and duplicate-resource identification. |
| [Bandukids/DLSS-Runtimes](https://huggingface.co/Bandukids/DLSS-Runtimes), [Bandukids/ComfyUI-DLSS5-VideoSuite](https://github.com/Bandukids/ComfyUI-DLSS5-VideoSuite), [t8star/DLSS-NR-Comfy](https://huggingface.co/t8star/DLSS-NR-Comfy), [T8mars/comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) | Runtime/package inventories and wrapper/application comparisons. |
| [dashengAi runtime](https://huggingface.co/dashengAi/ComfyUI-NV-DLSS-Frame-dashengAi-runtime), [dashengAi-Q/ComfyUI-NV-DLSS-Frame-dashengAi](https://github.com/dashengAi-Q/ComfyUI-NV-DLSS-Frame-dashengAi), [Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation](https://github.com/Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation) | Native-wrapper activation, input contracts, transfer/IPC cost and frame-interpolation organization. |
| [lisitskyaa/ComfyUI-DLSS5-NR](https://github.com/lisitskyaa/ComfyUI-DLSS5-NR), [HECer/ComfyUI-DLSS5](https://github.com/HECer/ComfyUI-DLSS5), [aiplatforms-ru/ComfyUI-DLSS5-Neural-Rendering](https://github.com/aiplatforms-ru/ComfyUI-DLSS5-Neural-Rendering), [DaniilSokolyuk/video2dlssnr](https://github.com/DaniilSokolyuk/video2dlssnr) | Host wrappers, media inputs, temporal lifetime and application-level measurement boundaries. |
| [taowen/dlss5-as-inpainting](https://github.com/taowen/dlss5-as-inpainting), [danielblnc/DLSS-NR-on-AMD](https://github.com/danielblnc/DLSS-NR-on-AMD), [ryusuzaku/DLSS5FSR](https://github.com/ryusuzaku/DLSS5FSR) | Alternate application contracts, AMD/HIP reconstruction and offline-oracle approaches. |
| [Dagherbou/OptiScaler_DLSSNR](https://github.com/Dagherbou/OptiScaler_DLSSNR), [aiertana/wwmi-dlss5-kit](https://github.com/aiertana/wwmi-dlss5-kit), [LagazYT/DLSS-5-Definitive](https://github.com/LagazYT/DLSS-5-Definitive) | Loader/forwarder behavior, game integration, runtime inventories and feature-routing options. |
| [machiavelli31/DLSS-AiO-Swapper](https://github.com/machiavelli31/DLSS-AiO-Swapper) | Runtime import/A-B management and component-source policy. |

## Maintenance

When adopting another upstream implementation, add its repository, exact source
location, recorded revision when available, license/notice location and a brief
description of local changes. Update the affected project's README as well.
Keep research-only credits separate from direct source imports, and preserve
original copyright/license headers when code is copied or adapted.
