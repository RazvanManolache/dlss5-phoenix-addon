# Feature catalog

[Project overview](../README.md) · [Documentation index](README.md)

The full feature list and route-dependent options. Start with the [user guide](USER_GUIDE.md) for the overlay and [compatibility guide](COMPATIBILITY.md) for execution choices.

**The controls got a makeover. Your sliders survived.** Four tabs, colored cards,
live Look controls and useful details that expand when you need them.

**Old APIs, fewer surprises.** SDR RGB10A2 capture, legacy DirectInput,
late-loaded input hooks and temporary renderer probes get dedicated handling.

## Features

- **Choose the instruction path:** Automatic prefers the GPU's native kernels;
  style-enabled experimental bundles also offer RTX 20/30/40 fallbacks on newer
  cards. Same models, real-target binaries, no older-GPU performance simulation.
  [Compatibility styles](NATIVE_ARCHITECTURES.md#selectable-older-instruction-styles).
- **Give the fallbacks less busywork:** aligned tile transfers and cheaper FP8
  decoding speed up the experimental older instruction paths while preserving
  their tested outputs. [1080p measurements and older-device limits](NATIVE_ARCHITECTURES.md#compatibility-optimization-check-at-1080p).
- **Another round of useful tinkering:** cached weight-side decoding, later
  Ada copy waits and wider reads cut full-model GPU time further: about 8–13%
  for 20/30-style, 4–5% for 40-style and 1–2% for native 50-style in the latest
  RTX 5080 fixture. The spill fix also saves memory. We tried the clever-looking
  alternatives too; the slower ones didn't make the cut.
  [What survived, what didn't, and the measurements](NATIVE_ARCHITECTURES.md#further-tuning-full-model-confirmation).
  [The 720p–8K sweep](NATIVE_TUNING_RESOLUTIONS.md) checks those gains across
  six more frame sizes, including 2560×1080 ultrawide, with the small native-path
  differences and mixed pairs left visible.
- **Keep the pixels moving:** native D3D9 readback uses system-memory surfaces;
  SDR RGB10A2 capture supports D3D10/11/12. HDR capture remains guarded.
- **Survive the warm-up act:** temporary renderer probes can give way to the
  real ReShade provider without leaving callbacks aimed at the old one.
- **Old-school controls included:** the optional Phoenix Input companion
  handles DirectInput 7/8 A/W, buffered polling and late window discovery.
- **Catch the same pixel twice:** [Save original + modified](PAIRED_SCREENSHOTS.md)
  exports matching PNGs from one processed frame. Click it in Play or assign a
  hotkey in Setup; no extra neural pass required.

- **Useful first, detailed when needed:** Play / Image / Performance / Setup,
  a fixed Save button, measured status tiles and expandable cards. The x86 and
  x64 add-ons share the same controls.
- **Switch models without collecting old work:** transactional replacement,
  clearer pending/loaded model feedback and retirement of unused GPU buffers.
- **Optional background-input companion:** a separate [Phoenix Input](PHOENIX_INPUT.md)
  download for local keyboard, mouse and virtual-controller automation. Your
  desktop cursor can keep its day job; the rendering addon does not need it.
- **Two sizes of neural rendering:** Original and the faster [Lightweight FP8](SMALL_FP8.md).
- **RTX 20 / 30 / 40 / 50:** an optional **RTX Compatibility** package includes
  architecture-specific kernels for both models. Older RTX 20/30/40 paths are
  experimental. **I don't own a 20/30/40-series card:** I built their kernels,
  but my NVIDIA hardware testing is on an RTX 5080. The [synthetic instruction-style tests](PERFORMANCE.md#built-for-older-rtx-cards-measured-on-rtx-5080)
  exercise those paths on that card; they don't simulate older-card performance.
- **Pick your renderer:** Phoenix DLSS5, NVIDIA DLSS5 or side-by-side comparison.
- **Tune the look live:** tone, structure, skin, masks, intensity and appearance
  styles, without restarting the model for every slider movement.
- **32-bit and 64-bit games:** matching ReShade add-ons, eligible same-process
  execution and an external x64 helper. Software-rendered games can use the helper.
- **Bring your weights, not another DLL copy:** compatible NVIDIA model DLLs,
  native model folders and the [shared model library](SHARED_MODELS.md).
  The pinned OpenNR FP16 compatibility DLL is also recognized.
- **Your paths count:** configurable model/runtime locations, checked DLL
  architecture, package hashes and driver profiles.
- **Automatic route selection:** eligible internal routes first, helper fallback
  when needed, and explanations for unavailable choices. In-process is not a speed guarantee.
- **Compare the same frame:** input, enhanced, split and difference views, plus
  diagnostics for processing time and execution location.
- **Broad graphics capture:** D3D9/10/11/12, OpenGL and Vulkan integration, with
  API- and architecture-specific limits in the [compatibility guide](COMPATIBILITY.md).
- **Less housekeeping:** cached schedules, lazy appearance buffers, lower startup
  peaks and conservative defaults for alternate backends.
- **Build your processing chain:** reorder neural rendering, scaling, color
  treatment and optional two/three-scale fusion. Disabled stages bypass without
  extra neural work; each active neural stage owns its own temporal history.
- **Motion when the game has none:** optional NVIDIA Optical Flow, engine-vector
  priority, confidence rejection and scene-cut history resets on eligible routes.
  Estimated motion is not engine-provided motion or invented depth.
- **Scale deliberately:** NVIDIA Image Scaling and an optional experimental tiny
  2x network, with separate processing/output sizes on the shared D3D12 pipeline.
  The tiny scaler is not the Lightweight DLSS5 model and needs external weights.
- **Watch the whole pipeline:** optional monitoring, coordinated latency budgets,
  last-working-settings recovery, comparison zoom/loupe and screenshot receipts.
  Unsupported route controls explain their limits instead of silently doing nothing.
- **Give crowded x86 games some breathing room:** streamed model authentication,
  selected-resource reads, bounded weight staging and address-space checks before
  in-process startup. Automatic portable selection prefers a validated GPU.
- **Use it outside a game:** native SDK and `phoenix-process` file/framed-stream
  interfaces, with fixed offline quality and protection against overwriting results.

These pipeline additions are route-dependent, not promises for every capture
hook. [Processing-chain guide](PROCESSING_CHAIN.md) explains what runs
where, costs and experimental limits. VSR and OptiScaler remain deferred.

## What you can play with

**Tune the look.** Tone, structure, skin and guide controls, without leaving the game.

![Tone, structure, skin masking, motion scale and depth controls](features/appearance.png)

**Keep your comparisons.** Timing rows survive backend and settings changes.

![Retained GPU timing history with a recent timing graph and separate Phoenix DLSS5 and NVIDIA DLSS5 columns](features/timing-history.png)

The values above are from the screenshot session, not the controlled benchmark.

- **Choose the runner:** native RTX, CUDA, Vulkan, HIP, CPU or SYCL in the Windows addon, with experimental Metal in the standalone Apple runtime, subject to
  the installed provider and device capabilities. Compiled runners need no
  Python or Torch at runtime. The native bundle currently targets **`sm_120`**;
  this is not a claim of support for every RTX generation.
- **Choose the processing device:** use the game GPU or select another available
  device for a portable runner. In principle, one GPU can render the game while
  another runs neural rendering—the same division of work used in
  [NVIDIA's early two-GPU showcase](https://allthings.how/dlss-5-requires-just-one-rtx-gpu-the-dual-5090-demo-was-a-tech-showcase/).
  Phoenix DLSS5 has an experimental offload path; real dual-discrete-GPU performance
  is unverified, and transfer cost/result age matter. The NVIDIA DLSS5 DLL route stays
  on the game's rendering GPU.
- **Choose the model source:** browse to a supported NVIDIA DLSS5 DLL or compatible
  model folder, then validate loading and a small inference run. Models and the
  NVIDIA DLSS5 DLL are supplied separately.
- **Share the large NVIDIA DLL:** set **NVIDIA DLL search folder** in the full addon
  panel. Local files still take precedence; x86/x64 and helpers can use the shared
  library without duplicating it. [Search order and configuration](MULTI_MODEL.md#shared-nvidia-dll-location).
- **Choose resolution and upscaling:** Normal or reduced-resolution neural
  rendering with processing quality and aspect controls. Upscaling choices are
  built-in spatial reconstruction, NVIDIA DLSS Super Resolution and Intel XeSS,
  for both Phoenix DLSS5 and NVIDIA DLSS5 NR output. Temporal upscalers run on the game GPU
  and require valid depth/motion guides and separately supplied vendor runtimes.
  Unsupported configurations show a spatial fallback. Gameplay quality is
  still unverified; XeSS has been exercised on NVIDIA DLSS5, not Intel hardware.
- **Tune the look:** local tone and structure, skin masking, motion/depth guides,
  and route-specific NVIDIA DLSS5 intensity/style/exposure controls.
- **Scale neural work dynamically:** enable **Enable approximate network** and
  **Adaptive blocks**, then choose **Phoenix DLSS5 GPU time** or **Desired game
  FPS**. Phoenix DLSS5 reduces the number of executed transformer blocks under load
  and restores them as performance allows, within your minimum and maximum.
  Time-budget mode uses measured inference GPU time; FPS mode uses the game's
  presentation rate. Reduced depth trades image quality for speed and is an
  experimental approximation. A target is not a contract with physics.
- **Tune the workload manually:** experiment with transformer-block repetitions
  and refinement passes to explore the quality/performance tradeoff.
- **Compare before committing:** split-screen and difference views, swappable
  sources, presets, history resets and timing diagnostics.
- **Keep the practical bits practical:** automatic CPU worker count or a manual
  limit, visible model-loading errors, and a Save configuration button at the top.
- **Keep installation compact:** runner libraries and kernels can be embedded
  in one addon; they unpack to a private runtime cache as needed. The NVIDIA DLSS5
  model DLL stays separate.
