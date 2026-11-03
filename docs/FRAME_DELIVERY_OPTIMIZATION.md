# Frame delivery and worker overhead

Game presentation FPS, newly completed processed frames, model GPU time and
worker wall time measure different things. Re-presenting one completed image
does not count as another inference. Never infer delivery FPS from the reciprocal
of a worker timing.

## Capture scheduling

A single outstanding GPU snapshot can turn a two-presentation capture fence lag
into one new processed input every other presentation, even when the worker is
fast enough. DX9 now uses a bounded FIFO with up to two pending snapshots.
Other graphics APIs retain one slot because their existing staging resources
are shared. This is an overlap optimization, not a reduction in neural quality.

Each pending entry retains its own fence, source-frame token, timestamp and GPU
texture slot. Retirement alone does not release that texture: submission must
transfer it into the model-input role first. Completed output is composed with
that exact input. Screenshots use the same frame token, rather than the most
recently captured image. Slots are not overwritten or reordered.

The second pending DX9 texture is optional. Allocation failure keeps the original
single-slot route; absence of the first optional texture also disables capture
overlap with a busy worker. The overlap path allocates up to two additional RGBA8
images (about16MiB total at1080p, with driver allocation overhead varying). Queue depth
is fixed, so a slower worker cannot grow an unbounded backlog. No new inference
wait or drawing is added after Present.

Telemetry exposes `capture_slots`, `pending_captures`, retirement/submission
counters and source/output frame lineage. The CPU scheduling regression models
two-presentation capture latency. The isolated DX9 test checks two separately
colored queued inputs, slot reuse and both corresponding comparison images.
These checks do not replace an actual-game FPS retest.

## Worker submission

The optional SDK output-postprocess contract lets the native worker record its
RGBA8 packing and GPU readback in the inference command list. The SDK's final
fence covers that work. The worker then copies completed bytes to its caller
without issuing another output command list and CPU fence wait. The upload
allocator is still retired before reuse, including on evaluation failure.

The new exports are additive. Older SDKs and graph/portable executors keep the
existing separate output submission. The worker queries actual session support;
it does not assume that a loaded DLL implements the optimization. Appearance
controls still use the advanced path, while neutral controls retain the simpler
executor. Model GPU timestamps end before the output callback, preserving their
existing scope. Worker/transfer wall time includes that output work.

Callbacks may only record commands. They cannot submit, wait, re-enter the SDK or
throw across the DLL boundary. The software D3D12 test verifies that callback
readback bytes are available when the SDK call returns and checks the debug
layer for resource-state errors.

The x86 caller also caches unchanged pipeline-control JSON and serializes the
frame parameters once, avoiding the former serialize/parse/serialize cycle.
Live pipeline edits invalidate the cache normally.

## Reproduction

- `python tools/test_route_optimization.py`: x86/x64 queue regressions and
  model-free software D3D12 SDK/fence checks.
- `python tools/test_route_optimization.py --gpu-dx9`: additionally validates
  real DX9 snapshot slots and comparison pairing in an isolated hidden window.
- `python tools/profile_wow64_worker.py --runtime PATH --source SOURCE.json --output RESULT.json`:
  opt-in1080p native-library profile,16 warmup frames and32 measured frames per
  monitoring mode. It verifies the requested model, sustained motion and output
  SHA-256. Model sources remain external; no games are launched or modified.

Detailed stage timings add instrumentation overhead. Compare ordinary-mode wall
time separately, retain both distributions and check output hashes. These
synthetic measurements must not be presented as game FPS or clean native-game
baselines.

## Isolated1080p comparison

RTX5080, native x64 library inside an isolated x86 caller, both original and
optimized runtimes alternated in G/H/H/G order. Each run used16 warmup frames
and32 measured frames per monitoring mode. Values below average the two run
medians for each runtime. Both models retained identical output SHA-256 across
all compared runs and monitoring modes, with sustained valid motion.

| Model | Monitoring | Previous worker ms | Optimized worker ms | Observed change |
|---|---|---:|---:|---|
| Original | Ordinary |7.32 |7.29 |Effectively unchanged; overlapping run ranges |
| Lightweight FP8 | Ordinary |5.67 |5.51 |About3% lower in this sample |
| Original | Detailed |7.64 |7.55 |About1% lower; overlapping run ranges |
| Lightweight FP8 | Detailed |5.54 |5.58 |About0.6% higher; overlapping run ranges |

These are modest worker changes, not a large neural speedup. Detailed
`evaluate_and_wait` now includes batched GPU packing; `pack_readback` becomes
mostly the final CPU copy. A smaller latter stage must not be mistaken for an
equal reduction in end-to-end time. The bounded capture queue targets a separate
presentation-sized delivery delay. Its actual-game benefit still requires a
fresh game retest; the previously paused suite was not resumed for these checks.

## Where the overhead goes

These measurements identified the bottlenecks **before** the queue-ordered
motion, compact status and direct byte-buffer input changes described below.

An additional isolated1080p diagnostic measured240 frames per case, with one
fresh GPU timestamp sample every30 frames (eight samples per case). The x86
caller uses the internal x64 library, estimated motion at180 analysis rows,
ordinary full-quality models and compact RGBA8 transport. These are diagnostic
means, not game FPS. No game was launched or modified.

| Worker component | Original ms | Lightweight FP8 ms |
|---|---:|---:|
| Model GPU interval, including final output copy |3.77 |2.26 |
| Motion preparation, dependencies and conversion |1.29 |1.30 |
| CPU input staging/enqueue and completed output copy |0.91 |0.85 |
| SDK status generation/parsing and caller status decode |0.60 |0.52 |
| Recording model/output commands on the CPU |0.34 |0.22 |
| Remaining GPU completion interval beyond model timestamps |0.44 |0.38 |
| Submission, controls, bookkeeping and request/response remainder |0.34 |0.28 |
| **Total worker call** |**7.70** |**5.81** |

Rows are rounded. GPU intervals and CPU waits overlap: the model interval is
subtracted from the corresponding completion wait above, not added twice. That
remaining interval includes untimed input canonicalization/output packing and
completion delivery; it is not a measurement of pure scheduler overhead. Model
GPU means use fresh samples only, so that subtraction is approximate.

Motion preparation consists mostly of waiting for the conversion submission:
about0.92ms for Original and0.98ms for Lightweight. Provider submission costs
another0.23–0.27ms; conversion recording/submission and queue synchronization
make up the remainder. The CPU waits here before recording the neural commands,
despite an existing GPU queue-ordering chain. The optimized direct SDK owner
now records the model commands while that conversion remains queued, retaining
the final consumer fence and error-path resource lifetime.

The worker calls the full SDK status API every frame. That builds memory
accounting, tuning and pipeline JSON which the worker parses and mostly drops.
Generation plus parsing alone measured0.44–0.51ms. The additive compact
per-frame status API now avoids that work, retaining full diagnostics on demand
and falling back to the full API with an older SDK. The
remaining native request/reply and report-encoding interval measured roughly
0.08–0.10ms; it is not the dominant source of overhead.

The native route tested here still stages CPU pixels into D3D12 and copies the
completed output back to CPU memory. Being in the same process does not make
that transport GPU-direct. These worker timings also exclude the game's own
capture and final return/upload.

Collecting GPU timestamps on **every** frame added about0.4ms of timestamp
readback bookkeeping; sampling every30 frames reduced its mean to0.014–0.016ms.
The table uses the sampled run. Detailed CPU clocks and status fields still add
instrumentation overhead, so these totals should not replace ordinary-mode
performance measurements.

The legacy direct `complete_call_ms` timer starts after motion preparation.
Its established clock boundary and configured adaptive-budget policy remain in
place. With ordered motion, pending GPU work can contribute to the final wait;
the elapsed value should not be interpreted as pure neural time. Detailed monitoring
now exposes exclusive `frame_stages_ms` intervals and a `total` that includes
motion, plus `motion_stages_ms` and worker status/transfer intervals. This makes
the timer boundary explicit. The comparisons below use fixed quality with
adaptive quality budgets disabled.

## Verified worker optimizations

- **GPU-ordered motion:** direct SDK sessions keep motion conversion on the
  same ordered queue and retain its resources until the final model consumer
  fence completes. They do not wait on the CPU between conversion and neural
  command recording. Allocator reuse tracks the last conversion fence, rather
  than an unrelated newer producer hand-off. Other owners retain the
  conservative CPU-retired behavior unless they satisfy this lifetime contract.
- **Compact frame status:** the worker requests motion, applied controls and
  timing without rebuilding memory inventories each frame. Full SDK status
  remains available; older SDKs use the compatible existing path.
- **Direct GPU input conversion:** a compute shader reads tightly packed RGBA8
  upload bytes directly into the model's half-float texture. It removes an
  intermediate RGBA8 GPU texture, its copy and an unused adapter output buffer.
  This saves approximately two RGBA8 images (about16MiB at1080p, subject to
  allocation granularity). Half rounding, source alpha and invalid-output
  rejection retain the established pixel contract. CPU staging remains for
  legacy routes that cannot share textures.

An isolated RTX5080 x86/internal-x64 comparison alternated the previous and
optimized runtimes in A/B/B/A order. Each runtime had two runs per model,
with16 warmup and32 measured frames per monitoring mode. Ordinary-mode values
below average the two run medians:

| Phoenix DLSS5 model | Previous worker ms | Optimized worker ms | Reduction |
|---|---:|---:|---:|
| Original |7.70 |6.70 |13.0% |
| Lightweight FP8 |5.71 |5.04 |11.8% |

Detailed monitoring measured10.2% and9.8% reductions respectively. These are
whole worker-call improvements, not faster model kernels or measured game FPS.
Both models retained the same complete output SHA-256 across all A/B runs and
both monitoring modes. With scene detection off, optimized samples reported
GPU-ordered motion and zero motion CPU waits. Compact status generation plus
parsing measured about0.08ms. Scene-cut thumbnails still require their small
explicit CPU readback when enabled.

The safety checks cover exact GPU half conversion, all finite half output
quantization, source alpha, nonfinite-output refusal, compact/full status
agreement, hardware motion, and draining after an error with motion work queued.
The paused game campaign was not resumed for these isolated checks.
