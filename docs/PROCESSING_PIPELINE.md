# Explicit processing and model-aware UI

<!-- current-two-model-evidence -->
See [the current support matrix](SUPPORT.md) for model/backend and frontend-specific availability, and [the current benchmark](benchmarks/current/README.md) for isolated neural/scheduling and portable V3 costs. Those measurements do not time every addon conversion, reconstruction or insertion-stage combination; the original verification boundary below remains applicable to those integrations.
<!-- /current-two-model-evidence -->

Phoenix now exposes the controls recovered from RenoDX 0.2026.1016.1704 through its own independently authored processing code. The recovered DLL and earlier extracts are untouched. This is an implementation of the controls, not a claim that Phoenix duplicates RenoDX's addon shaders pixel for pixel.

## Model selection

NVIDIA neural model selects Match Phoenix architecture (default), DLL default, Original (preset 1), or Lightweight FP8 (preset 2). This create-time setting recreates the vendor feature after queued frames retire. The selected, authenticated external DLL takes precedence over a game-local copy. Changing the vendor DLL after it has been initialized requires a restart; changing presets within that DLL recreates features normally. Lightweight is refused if the loaded DLL lacks WEIGHTS_CC_SMALL_FP8. The DLL default remains explicitly version dependent. Artistic Style A/B/C is separate from the architecture preset.

The Phoenix picker reports Original's 71 learned blocks / 658 tensors / 153 records / 154 native launches and Lightweight's 31 / 302 / 77 / 78. Both retain 8 central transformers. These launch counts describe the neural graph, excluding surrounding clear/copy and processing work. The experimental x86 driver-host route has an Original-only launch ABI; its UI and runtime reject Lightweight and direct users to the x64-library/helper routes. Other native RTX and portable routes retain both models.

## Granular controls for both Phoenix models

Open **Phoenix DLSS5 experiments**. The selected model uses the same depth, repetition, and adaptive controller settings; these controls are Phoenix execution changes, separate from NVIDIA's architecture preset.

| Property | Original | Lightweight FP8 |
| --- | --- | --- |
| Central transformers | 8, learned graph IDs 31–38 | 8, learned graph IDs 11–18 |
| Native neural launches outside those transformers, per pass | 114 | 38 |
| Native neural launches per retained transformer repetition | 5 | 5 |
| Depth ceiling / minimum adaptive depth | 1–8 / 1–ceiling | 1–8 / 1–ceiling |
| Each transformer's repetition count | 0–4 | 0–4 |
| Whole-network processing passes | 1–10 | 1–10 |
| Adaptive target | Measured Phoenix GPU milliseconds or game presentation FPS | Measured Phoenix GPU milliseconds or game presentation FPS |

Zero repetitions bypass that transformer and forward its input. Depth keeps a prefix of the selected model's eight transformers, and repetition counts apply only inside that prefix. Adaptive changes retain those counts and jump past disabled tail blocks when changing depth would save no work. At the configured minimum, fixed encoder/decoder work, repeated network passes, and surrounding processing can still exceed the time target. The GPU budget is a feedback target, not a hard deadline that interrupts inference; whole-game FPS can also be limited by other work or a frame cap. Skipping, repeating, or truncating learned blocks is an untrained approximation.

Timing comes from the selected configuration, rather than scaling Original's timings by a launch-count ratio. Pending measurements are tagged with the controller's observation epoch and settings, so late results cannot influence a different model/settings window. Stage timing stores all fifteen groups across all ten passes for either native graph.

Adaptive GPU timing uses completed backend GPU events for portable workers, and native queue timestamps on the RTX route. Portable final-frame frontends enable GPU budgets when the selected GPU and updated runtime advertise the capability. CPU wall time and SYCL CPU queue events are excluded from GPU budgets. Desired FPS additionally requires the actual application's presentation cadence; companion hidden-window FPS is not used. The Original-only experimental x86 driver host retains its fixed-network restriction. See [PORTABLE_MODEL_CONTROLS.md](PORTABLE_MODEL_CONTROLS.md) for Metal, Vulkan, HIP/AMD and SYCL integration.

## Processing controls

Existing profiles keep their legacy bridge. Open **Processing pipeline** and enable **Explicit scaling and color controls** to select the new path. Enabling it selects spatial reconstruction and clears the legacy reference-grid unjitter option. Legacy resolution/upscaler controls are disabled while explicit scaling owns the working extent.

| Control | Runtime behavior |
| --- | --- |
| Hook method | Off bypasses processing; Render modifies the observed DLSS color input before SR/RR; Upscaled processes its output; Present processes the current swapchain image before ReShade effects/UI. Nested NGX wrappers are processed once. |
| Present guide reuse | Never uses zero motion and resets history each frame. Optional uses captured current-frame motion/depth when available, otherwise the same spatial policy. Required withholds processing when those guides are absent. |
| Resolution scale | 0–200% per axis, with a 16-pixel minimum canvas and the backend's maximum extent. NR itself always receives equal input/output dimensions. Minimum/clamped dimensions can change the working aspect; reconstruction restores the output extent. |
| Scaling domain | scRGB-nl, Signed PQ, or linear BT.709 filtering; source transfer and gamut are decoded before filtering. |
| Downsample | Exact footprint area integration, bilinear, or nearest. |
| Upsample | Bilinear, Catmull-Rom bicubic, or nearest. |
| Reconstruction | RGB residual adds the resampled linear RGB change to the full-resolution original; Processed image resamples the complete result. Original alpha is retained. |
| Input encoding | Auto, Linear BT.709, sRGB, HDR10 (PQ/BT.2020), scRGB (80 nits/unit), or signed scRGB-nl. |
| NR signal | sRGB, PQ SDR, or linear SDR. This applies to both Phoenix and NVIDIA feeds and inverse conversion. |
| Diffuse white | Automatic host value when supplied and in range, otherwise 100 nits; explicit override 48–500 nits. |
| Exposure | 0–100x saved/UI range; effective exposure is clamped to 0–10x before optional inversion. Off uses unity. Zero exposure preserves the original on inverse conversion to avoid undefined division. |
| UI correction | Auto follows the supplied host policy; Off disables; On requests correction. Phoenix requires explicit UI/alpha coverage and refuses forced On without it. |
| Block Hudless | While Present enhancement is active, suppresses the Streamline HUD-less color tag before forwarding it. It does not alter ordinary game resources. |
| Processing passes | 1–10. Native Phoenix and NVIDIA passes each own separate temporal histories. Color-only transport resets history; portable runtime retains its per-pass history array when real guides are supplied. |

The portable/x86 pixel transports use the same independently authored color equations and reconstruction on CPU around inference. They do not turn a color-only source into a temporal source. All settings travel through the existing INI/preset/control bridge/helper serialization.

## Source contracts

Game insertion stages need an observed D3D12 DLSS feature and resource states. Render uses the feature's color input; Upscaled uses its output. Motion can be resampled from a different render extent. Final-frame-only frontends support Present and Off; Render/Upscaled requests produce an explicit status. External feeders choose their own insertion timing, so their hook selector is disabled.

Present takes owned guide snapshots on the observed game queue, then requires the same queue, output extent, and presentation interval at reuse. Source resources remain retained until their recording retires. Unsubmitted/reset lists abandon their snapshots. Scalar R32_FLOAT, R32_TYPELESS and D32_FLOAT depth are copied into owned, readable R32 snapshots before resampling; fully typed depth is not bound as a float SRV. Missing, unobserved, multi-plane or otherwise unsupported depth causes Optional fallback or Required withholding. Present's owned command list and conversion buffers use the existing synchronous D3D12 bridge and completion proof. Game Present uses Native RTX and writes back after the owned submission retires. Its UI disables that stage for portable game-DLSS executors. Portable Present remains available through the final-frame frontends/helpers, with their existing coherent frame transport.

Auto encoding uses actual swapchain color space at game Present, the external feeder's declared transfer for feeder input, and linear BT.709 for observed game render/upscaled color. Explicit encoding overrides this inference. Byte-only compatibility transports retain their existing SDR/format restrictions; an encoding dropdown does not remove those transport limits. Forced Phoenix UI correction also retains the existing explicit appearance/coverage contract; no UI mask is invented.

## Verification boundary

Focused validation is recorded under `artifacts/current/processing`. It covers 32/64-bit settings and helper serialization, model/preset mappings, resolution bounds, signed transfer/gamut round trips, exposure edge cases, residual detail preservation, real ImGui widget interactions, shader compilation, addon compilation/linking and related CPU contracts. No game, graphics device, GPU inference, timing comparison or new visual parity test was run for this change. Earlier four-model parity receipts cover the earlier rendering recipe and are not evidence for every new option combination. Installed games, extracted research, model DLLs and the frozen reference were not replaced.
