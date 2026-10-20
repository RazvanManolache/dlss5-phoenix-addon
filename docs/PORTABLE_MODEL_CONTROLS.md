# Both models on portable backends

<!-- current-two-model-evidence -->
Fresh device benchmarks now exercise both models on CUDA, NVIDIA Vulkan, AMD Vulkan and CPU; the CUDA/NVIDIA Vulkan profiles also measure depth, skip/repeat, 2/4/10 passes, GPU-budget adaptation and V3 appearance inputs. See [current measurements and limits](benchmarks/current/README.md). Metal/HIP/SYCL retain source/graph validation only in this refresh.

RTX Vulkan measurements use `vulkan_fused_feed_forward: false`: the automatic fused FFN failed 4K reset repeatability in both models. The default optimized-path issue remains unresolved; full trained execution through the ordered pair is the checked workaround.
<!-- /current-two-model-evidence -->

Original and Lightweight FP8 use the shared portable decoder, graph and frame runtime on Metal, Vulkan, HIP/AMD, SYCL, CUDA and CPU. Backend operators consume the recorded graph rather than a fixed Original-model launch schedule. No proprietary DLL, weight or prior extraction was replaced.

Select `weight_source.model` as `original` or `small-fp8` in the existing runtime configuration. Metal keeps its Apple silicon/unified-memory requirement; Vulkan keeps its capability-checked variants/fallbacks; HIP and SYCL keep their existing SDK/device requirements. The experimental x86 native driver-host ABI is separate and remains Original-only.

## Frame scheduling

The existing C frame API accepts the following optional metadata for either model and every portable backend:

```json
{
  "transformer_blocks": 8,
  "repeats": [1, 1, 1, 1, 1, 1, 1, 1],
  "passes": 1,
  "adaptive_budget": {
    "enabled": true,
    "mode": "gpu_time",
    "target_ms": 5.0,
    "minimum_blocks": 2
  }
}
```

`transformer_blocks` is the 1–8 depth ceiling. Each of the eight repetition counts is 0–4; zero forwards the block input. Counts outside the selected prefix are not executed. The eight controls map to Original's learned blocks 31–38 or Lightweight's 11–18. `passes` is 1–10; each pass has its own temporal history. Omitting the new fields retains the existing eight-transformer, caller-supplied repetition schedule and single-pass defaults.

Adaptive GPU time sums backend GPU events across all passes of this runtime frame call. Metal submission timestamps and Vulkan/HIP/SYCL GPU event intervals use this same contract. CPU wall time, CPU queue events, shader/graph construction, host transfers outside the timestamp interval and another backend's timing estimates do not become GPU-time samples. Missing GPU timestamps leave adaptation awaiting observations. The target is feedback, not a hard inference deadline. Fixed model work or a configured minimum can prevent reaching it.

For application FPS, set `mode` to `presentation_fps`, set `target_fps` (10–360), and provide `presentation` with an unsigned monotonic `timestamp_us` and an explicit boolean `active` on each frame call. These timestamps must come from the application's real presentation cadence. An offscreen helper's iteration rate is not application FPS. Pauses/inactive windows reset observations. The Windows direct addon already owns its game-present controller; portable final-frame/helper UIs expose GPU-time adaptation and keep their FPS target unavailable.

The runtime returns `network_schedule` with the depth used, next depth, observation estimate, GPU-event validity and measured GPU milliseconds. `phoenix_runtime_capabilities` advertises `granular_controls`; hosts should check this before enabling the new metadata against an older runtime. No C ABI layout or function signature changed.

## Worker and UI integration

The Windows direct addon consumes a worker completion once, using its recorded configuration epoch, model, depth and generation. Holding or reprojecting one image cannot produce extra samples. An image that expired under the delivery-age limit can still contribute its completed GPU cost if its configuration matches; its display eligibility is unchanged. Invalidated generations, old model/configuration samples and worker failures are rejected.

Portable final-frame frontends now forward their GPU budget and raw per-block repetition profile to the runtime, and show the depth and estimate returned by that runtime. The UI only enables this path for a GPU backend with the new capability. NVIDIA preset selection remains a separate vendor feature; these approximate scheduling controls change Phoenix execution.

## Verification

`artifacts/current/portable-models/validation.json` records CPU-only scheduler/worker-feedback tests, authenticated Original/Lightweight graph recording for Metal, Vulkan scalar/fused, HIP and SYCL feature profiles, generated Metal arithmetic simulated on the host, and 32/64-bit Vulkan adapter compilation. Graph recording checks every transformer control, skips and custom repetitions and refuses execution. Updated 32/64-bit addon/runtime development builds remain in `artifacts/current/processing`.

This Windows host did not run a Metal compiler or GPU, a HIP/SYCL compiler or GPU, Vulkan inference, games, benchmarks, or deployment for this update. Apple compilation and backend device execution remain to be verified on their respective platforms. Shared source, graph recording, host arithmetic and successful builds do not establish visual or timing parity on those devices.
