# Shared pipeline implementation

User-approved scope: native optical flow and addon controls; optional monitoring
and latency policy; initialization/startup rollback; optional scene detection;
NIS and VSR; comparison/export/subtitles; mod coexistence; media cache/jobs;
native CLI; editable chain in addon and player; optional multiscale fusion;
per-game driver settings; native tiny 2x upscaler; standalone player in
`../dlss5_video_player` integrated into Everything. OptiScaler integration is
explicitly deferred until Phoenix is satisfactory.
VSR was also explicitly deferred by the user on 2026-10-21 after the required
current NVIDIA SDK/runtime was found to be unavailable. Keep it unavailable in
the new recipe UI; do not substitute the older installed effects runtime.

Existing unrelated dirty files must be preserved. No game installation changes
or public publication are part of this implementation. Models remain external.

## Acceptance ledger

No feature is complete until its controls reach working processing code and its
relevant build and behavior checks pass. Source compilation is distinct from GPU
execution; 5080 tests cannot validate older cards. Update evidence below as work
finishes.

| Work | State | Evidence |
| --- | --- | --- |
| Shared pipeline options, contracts and scheduling | Native contracts and portable adapters implemented; current-build validation below | Shared recipe, independent stage histories, fixed offline policy and one coordinated native adaptive controller. Portable chains use one parent neural budget across branches, with child controllers disabled. The equal-extent portable C API keeps fixed pipeline priority; the shared D3D12 adapter owns the portable pipeline governor. Neutral chain/settings fold into the original executor. |
| NVIDIA motion provider and engine-vector priority | GPU translation and history-rejection checks passed; native and shared portable integration implemented | Actual 5080 NVOF active. Known 8x4 translation at full/half analysis heights: median endpoint error 0, p95 <=0.0625 pixel. Rejected vectors are moved outside the history image; trusted vectors remain bit-identical. Shared by the SDK and addon bridge. |
| Optional monitoring and pipeline latency policy | SDK/player source-to-Present feedback checked; physical display latency unmeasured | Monitoring off skips GPU queries and host transfer timers. Controller changes one permitted setting at a time, settles, and restores within requested limits. Player reports decoded readiness, processing and Present API return, excluding intentional pacing/pauses from adaptation. Reused paused frames do not replace the presentation-tail sample. Upstream capture/decode and display scanout are outside this measurement. |
| Durable initialization/runtime rollback | Addon, player and Everything recovery implemented; manual UI validation pending | Initialization failures restore working settings before retry. Interrupted trials, 30-frame health promotion, manual disable/restore/model selection and appearance changes have behavior checks. Recovery journal is separate from the installer journal. |
| Scene-discontinuity detection and reset fanout | Native shared detector and reset fanout checks passed | Optional scene detection; timestamp/source/dimension discontinuities reset the owning stream and all stage histories. Six independent stage instances tested across queued frames. |
| NIS provider | Native/addon reconstruction and 1080p host-cost checks passed | Odd 71x41 to 142x82 fixture with alpha, finite and constant checks. Explicit scaling stage. The current pipeline report also measures 960x540 to 1920x1080 NIS against spatial/tiny scaling; broader backend coverage remains pending. |
| VSR provider | Deferred by user | Current SDK requires NVIDIA sign-in; installed Video Effects 0.7.6 is insufficient for the proposed adapter. |
| Comparisons, receipts, subtitles and export | Direct renderer comparisons and native exports pass; player/addon controls connected | Same-frame reference, seven comparison modes without inference reruns. Player `artifacts/current/comparison-validation.json` checks both neural models, BGRA/NV12 and the window compositor; Split/Wipe select the correct independently captured pair members with zero byte error. A sampled 2560x1440 trailer frame through NV12 at a 1100x619 presentation target changes by mean 3.20/255. This is a sampled-frame check, not capture of a live player session. Addon saved-image event and player screenshot receipts include image hash; player records completed source identity. ReShade event has no exact source-frame association and labels that limitation. |
| Mod coexistence plans | Implemented conservative planner; UI validation pending | Everything inventories exact loader names and hashes, offers a free compatible proxy, preserves all mods. Three behavior tests pass; unknown chains require manual documented setup. |
| Persistent media cache and resumable jobs | Native jobs, live segments and cache maintenance integrated; GUI interaction not manually checked | Player `artifacts/current/phoenix-integration-validation.json`: both models and spatial 1x/2x, eight frames each. Interrupted/resumed and continuous decoded pixels match. Segment publication carries source timestamps; paired decoders synchronize at output geometry. Producer/read leases prevent active eviction, including two simultaneous readers. Clear/pressure eviction preserves unknown files, rejects linked paths and retries owned retired entries. Changed segments/carriers are refused and preserved. Native receipts do not claim Feature18 evidence. |
| Native CLI | Native RTX/spatial and shared portable framed streams implemented; packaged checks below | `phoenix-process` preserves pixels/timestamps and refuses existing outputs. The existing `phoenix_frame` RGB32F job interface also forwards the new chain metadata on CPU/CUDA and NVIDIA/AMD Vulkan, both models. Its output matches the public C API exactly and existing outputs are preserved. The shared SDK adapter also admits portable stages through the existing PXF1 framed stream; its offline policy fixes quality. |
| Editable processing chain in addon/player | Native execution checked; portable addon color/fusion source and capability gates checked | CPU color/scaling oracle, order changes pixels, disabled stages bypass exactly, invalid tuning preserves session. Player dialog applies transactional replacement and persists matching launch preferences. Updated portable runtimes advertise host-chain contract 1; the addon forwards metadata and rejects old runtimes explicitly. Those equal-extent restrictions apply to the portable C API path. The x64 asynchronous frame worker and player additionally use the shared D3D12 stage graph with independent processing/output extents, NIS/tiny scaling and pipeline frame services. Everything and the player select native RTX or installed native portable devices. |
| Multiscale fusion option | Native addon/SDK execution and player controls checked | Branches consume the same immutable source. A preceding neural prediction is reused only across pixel-preserving stages; standalone fusion includes its own full-size prediction. An affine fake network reproduced the previous double-processing bug before the correction. Independently rendered full/half predictions match both models' packaged fusion with zero error at 160x96. Eight blocks and block-8 repetition 0/1/2, preserved alpha, six standalone stage histories and queued reset fanout also pass. Useful-size observations remain fixture-specific. |
| Per-game driver settings and restoration | Transactions and read-only driver check passed; UI validation pending | Journal-before-save, inherited-value restoration, external-change preservation; transaction tests pass. Profile name/application count and shared-profile scope shown; ambiguous lookup refused. No actual driver settings changed. |
| Native tiny 2x upscaler | Native port, reconstruction oracle and 1080p host-cost checks passed; quality/cost remain experimental | Two odd extents on WARP and 5080 match an independent PyTorch CPU oracle with max RGB error <0.000487. Native importer/downloader checkbox and installed-only scaler selection. At 960x540 to 1920x1080, the current port takes about 58 ms including host transfers and produces pronounced texture artifacts on the tested face. Numerical agreement does not imply good image quality. |
| Native video player and Everything integration | Native checks and model-free packages passed; GUI interaction not manually checked | Both models, exact original/processed split and wipe, tuning, recovery and saved preferences. SDR/HDR direct rendering checked; current native export is SDR/CFR. Player ZIP includes relative SDK/kernel paths, pinned native helpers and notices. Everything's Release payload runs both models and matches the validated SDK and independent fusion oracle. External models are not bundled. RTX 20/30/40/50 kernel variants are included; physical execution was checked only on 5080. |

All deployed processing is native. Keep existing default recipes equivalent;
new optional stages must allocate lazily. Keep the native/portable ABI backwards
compatible. Interactive queues may discard superseded input; offline jobs may
not discard frames. Histories belong to stream and stage instance. Resume must
replay history or restore a verified complete checkpoint.

## Current validation checkpoint (2026-10-21)

The native SDK suite passed CPU contracts, WARP, hardware motion and confidence
consumption, both addon neural models, all eight block controls, native CLI and
the independent tiny-model oracle. Addon compilation passed 22 CPU suites and
two software D3D12 suites. The player and Everything Release builds succeeded;
eight targeted Everything pipeline/recovery/driver transaction tests passed.
Player direct comparison checks pass both models in BGRA/NV12, SDR/HDR, and
cached processing bypass. No desktop automation was used.

The model-free experimental addon package also passed its extraction/hash audit,
22 CPU suites, two software D3D12 suites and actual packaged-kernel execution of
both models at 320x180 and 1920x1080. It is not a gameplay compatibility test.

[Current pipeline measurements](benchmarks/current/pipeline/README.md) cover
46 fresh-process runs: two per recipe, both models at 1920x1080, two/three fusion
branches, two whole-network passes, four central blocks, 50% neural resolution,
180/360 optical flow and three 2x scalers. Timings include host transfers; process
DXGI memory is sampled after warmup, not a peak allocation figure. Native-pixel
face crops show smoothing from lower-scale fusion and artifacts from the tiny
checkpoint. Reset pixels repeat across both processes for every recipe.

Block-only initial tuning now folds into the established executor. Submitting
the identical tuning preserves that executor and its completed frame/history.
Both models' four-block/repetition recipe matches the established direct API
byte for byte in the packaged runtime. Later structural live changes can still
rebuild through the general chain; broader optimization remains separate work.

Portable color/fusion has 14 actual runtime cases using the released package's
backends, including deferred presentation, plus 24 byte-exact ordinary
reset/history/reset reference frames. CPU and CUDA use independent backend
instances per branch; Vulkan branches share one serialized context with separate
graphs and histories. Both x86/x64 portable bridge suites also check metadata
forwarding and explicit refusal of an older runtime. See
`artifacts/current/portable-pipeline/validation.json` and
`artifacts/current/portable-bridge-validation.json`.
The final experimental addon embeds that validated portable DLL and the matching
native SDK. It passed DLL loading, payload extraction, all packaged-content
audits and both models' 320x180/1920x1080 native-kernel checks. CLI checks against
the actual updated embedded runtime are recorded in
`artifacts/current/portable-pipeline/cli-package-validation.json`.

Build the complete current experimental payload with:

```powershell
python tools/build_experimental_rtx_addon.py --pipeline-runtime artifacts/current/portable-pipeline
```

This updates only disposable extracted staging and current experimental outputs;
the existing release remains the frozen reference. SDK/portable binaries now use
mapped compiler source paths and must pass the publication-content audit before
packaging. This is a local validated build, not a published release.

The follow-up integration audit corrected live attachment to prepare the pair,
metadata decoder and renderer before changing playback state. Four real player
cases cover BGRA/NV12 at 1x/2x with eight failure paths and eight compositor
readbacks. Cached results are not processed again. Requested output geometry and
timing mode travel with decoder swaps; prepared network decoders also honor
monitoring on their first frame. The player packager requires this current-build
receipt. Everything's target-FPS, maximum-time and minimum-block controls remain
available for both models with pipeline monitoring off; fifteen targeted UI,
pipeline, recovery, driver and mod-planning tests pass. These neural budgets are
independent of the optional pipeline governor.

Addon frame-service controls require the local final-frame D3D12 bridge or the x64 shared portable frame worker,
rather than treating every final-color source as capable. Experimental x86
native library/driver routes reject custom chains, motion estimation, scene
detection and pipeline monitoring explicitly. Both bitness route suites include
these refusal cases (383 checks each), alongside the existing bridge, policy and
IPC checks. This gates incomplete routes; it does not implement those services
on them or establish game presentation compatibility.

The shared portable SDK adapter closes the geometry/scaler, motion/scene,
pipeline-governor and player-backend gaps above. Each portable pass/branch owns
its native worker/history, avoiding provider singleton collisions during live
replacement. Parent D3D12 stages preserve alpha and explicit source geometry.
Unsupported optical-flow dimensions are refused before allocating driver
buffers, so the fallback retains a healthy device. Workers stop and drain
before their DLLs/staging are released.

Current-build receipts are `artifacts/current/pipeline-native/validation.json`
and `portable-pipeline-validation.json`; player receipts include
`portable-player-validation.json` and `package-runtime-validation.json`.
They cover CPU/CUDA and NVIDIA/AMD Vulkan, both models; independent C API pixel
oracles; geometry/color/NIS/fusion and separate whole-network histories;
monitoring-off block budgets; offline fixed quality; asynchronous addon worker
execution and retirement; BGRA/NV12/HDR comparisons; live backend replacement
and refusal preserving the working session; resumed/continuous/cache decoded
export equality; packaged file/stream source identities. Full GPU cases use
eight blocks, CPU one; export tests use eight source frames. These are small
fixtures, not real-time or gameplay claims. All D3D12 stages/presentation in this
matrix execute on RTX 5080, including AMD Vulkan neural inference.

Broader real-scene/backend measurements and manual frontend/game interaction
checks remain unverified. AMD-only presentation, physical older cards and
unavailable HIP/SYCL/Metal devices were not validated here. Raw insertion and
x86 native bridge routes keep their explicit capability limits. Generic portable
comparison/loupe edits no longer restart their worker or reset neural histories;
they appear with the next processed source frame. Player/native cached views
can redraw the completed pair without neural evaluation.

The final capture-worker check also enables neural processing for both models
and compares the rendered bytes against an independent public C API session,
including non-default tone/structure, intensity and style. Zero intensity
restores the input. Shared-frame routes expose the portable appearance controls;
older equal-extent routes reject them explicitly. Worker process IDs are checked
against live OS handles, and both original/current workers must exit on retirement.
SDK status and addon control exports distinguish the processing host from the
neural worker list; motion-enabled history is no longer labelled unconditional
spatial reset.

Everything invalidates processing-device discovery when the component root or
runtime manifest changes. It discards outdated completions, rechecks identity
before committing a result, retains available selections, and permits retry
after failure. The controls have an explicit refresh action and block launch
while discovery is pending. An offscreen WinForms regression exercises changed
packages, reversed completion order, in-flight replacement and failure/retry.

Source-to-Present telemetry does not measure physical displayed-frame age. HDR
export is outside the current SDR/CFR export route. VSR and OptiScaler remain
deliberately deferred. The acceptance table records implementation and measured
acceptance separately; it does not certify every backend/game combination.

## Publication checkpoint

The observations above describe their original development binaries. The
[current release](RELEASE_0.13.md) rebuilds the add-ons and affected runtime
companions with CPU/software-only validation at the user's request. No new GPU
inference or performance qualification is implied by publishing those packages.
See [release validation](VALIDATION_0.13.md) for the exact boundary.
