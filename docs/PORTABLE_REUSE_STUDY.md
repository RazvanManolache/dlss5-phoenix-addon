# Portable backend reuse study

These changes apply to **Phoenix DLSS5** portable execution. The native RTX instruction kernels and NVIDIA DLSS5 runtime are separate paths; their published timings are unchanged.

Measured on an RTX 5080, Ryzen 9 9950X3D and its integrated Radeon GPU. No older discrete AMD GPU, Mac, HIP SDK or SYCL SDK was available for this study.

## What changed

- AVX2 loads eight consecutive half values per row and transposes an 8×8 tile in registers. The matrix loop retains its ascending reduction/FMA order.
- AVX-512 retains each gathered pair of halves for the next reduction step instead of gathering it twice.
- Shared host frame preparation encodes uniform conditioning values once per frame. Spatial maps still override tone and structure per pixel; signed zero, padding and history inputs remain exact.
- A register-only CUDA FP8-to-half decoder was tested, then **discarded from production**: the measured GPU benefit was negligible. Its legacy ±480 codes deliberately differ from E4M3FN NaN codes. The verified prototype remains isolated in the CUDA math test.
- Vulkan tests aligned four-byte weight loads in the coalesced scalar fallback. Cooperative-matrix execution uses a different shader and already groups its weight reads.

These are bounded register/tile reuse changes. They add no expanded per-model weight cache, change no model parameters, and retain precision/publication boundaries.

## Whole-model measurements

Full Original and Lightweight networks, one pass, all eight transformer blocks, identical structured SDR input, fixed seed and zero motion. Each process warms independently, then checks reset, history and reset-again output hashes. Every paired control/candidate hash matched.

CUDA and NVIDIA Vulkan at 1080p and CPU/AMD at 320×180 use three alternating process pairs. The CPU/AMD 1080p scaling check uses **one pair only**: useful validation, not a stable performance estimate. Numbers below are full host-call medians, including preparation, transfers and resolve, rather than isolated GPU inference.

| Backend | Input | Model | Before (ms) | After (ms) | Host-call reduction | Process pairs |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CUDA | 1920×1080 | Original | 76.04 | 75.88 | +0.2% | 3 |
| CUDA | 1920×1080 | Lightweight | 57.01 | 54.38 | +4.6% | 3 |
| NVIDIA Vulkan | 1920×1080 | Original | 99.64 | 100.67 | -1.0% | 3 |
| NVIDIA Vulkan | 1920×1080 | Lightweight | 69.13 | 70.43 | -1.9% | 3 |
| CPU (32 workers) | 320×180 | Original | 687.78 | 626.69 | +8.9% | 3 |
| CPU (32 workers) | 320×180 | Lightweight | 332.54 | 313.29 | +5.8% | 3 |
| AMD Vulkan | 320×180 | Original | 616.06 | 612.49 | +0.6% | 3 |
| AMD Vulkan | 320×180 | Lightweight | 355.38 | 355.21 | +0.0% | 3 |
| CPU (32 workers) | 1920×1080 | Original | 8299.47 | 8338.68 | -0.5% | 1 |
| CPU (32 workers) | 1920×1080 | Lightweight | 5388.88 | 5094.85 | +5.5% | 1 |
| AMD Vulkan | 1920×1080 | Original | 10321.84 | 10108.65 | +2.1% | 1 |
| AMD Vulkan | 1920×1080 | Lightweight | 6032.83 | 6219.26 | -3.1% | 1 |
| AMD Vulkan (scalar fallback) | 320×180 | Original | 821.82 | 784.83 | +4.5% | 3 |
| AMD Vulkan (scalar fallback) | 320×180 | Lightweight | 463.91 | 443.65 | +4.4% | 3 |

A positive reduction means less time; small differences should be read against run-to-run variation. GPU event times and preparation times are retained in the [compact data](portable-reuse.json). Do not interpret noisy host-call differences as a GPU-kernel speedup. The CUDA candidate measurements include the decoder experiment that was subsequently discarded; production CUDA GPU execution remains unchanged.

## Isolated CPU work

Seven alternating timing pairs per resident matrix shape, with at least 40 ms per measurement. The operator test covers transposed/non-transposed FP8 and half weights, batch offsets, padded strides and incomplete row/column/reduction tiles. All 240 before/after layout checks are bit-exact.

| Loader | Reduction width | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| AVX2 | 32 | 0.2246 | 0.1504 | 1.49× |
| AVX2 | 128 | 0.6648 | 0.3698 | 1.80× |
| AVX2 | 512 | 2.3604 | 1.2252 | 1.93× |
| AVX2 | 1024 | 4.6288 | 2.3822 | 1.94× |
| AVX-512 | 32 | 0.1475 | 0.1369 | 1.08× |
| AVX-512 | 128 | 0.6707 | 0.4921 | 1.36× |
| AVX-512 | 512 | 1.9402 | 1.5273 | 1.27× |
| AVX-512 | 1024 | 3.6565 | 2.9441 | 1.24× |

The shared host input-packing test uses three process pairs and nine measured calls per process/case after warmup. All 12 packed-input cases match exactly, including odd extents, automatic masks, spatial maps and signed-zero controls.

| Input packing (auto mask on) | Spatial map | Before (ms) | After (ms) |
| --- | --- | ---: | ---: |
| 1920×1080 | No | 17.10 | 15.28 |
| 1920×1080 | Yes | 16.90 | 16.62 |
| 2560×1080 | No | 22.95 | 20.71 |
| 2560×1080 | Yes | 23.49 | 22.75 |

This is the CPU host feature packer. Backends that advertise device feature packing bypass it; it does not make their GPU network faster. Metal, HIP and SYCL can benefit when they use this shared host fallback, but no device performance claim is made for them here.

## GPU fallback experiments

Three alternating process pairs per shape, two warmups and at least 80 ms of measured resident matrix calls per process/shape. All outputs match exactly. These are isolated operations, not whole-frame latency.

The longer sampling window removed large outliers seen with the old three-call microbenchmark. CUDA was effectively flat (about 1.00–1.03×), so its decoder change was not promoted. The direct FP8 CUDA path already loads packed bytes.

| AMD scalar fallback matrix (M×N×K) | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: |
| 8192×128×32 | 0.877 | 0.878 | 1.00× |
| 8192×32×128 | 0.757 | 0.757 | 1.00× |
| 32768×512×128 | 47.759 | 41.963 | 1.14× |
| 130560×512×128 | 193.494 | 167.360 | 1.16× |
| 32640×1024×256 | 167.649 | 145.665 | 1.15× |
| 32640×256×1024 | 205.626 | 206.460 | 1.00× |
| 2048×1024×256 | 10.051 | 8.724 | 1.15× |
| 640×1536×512 | 9.321 | 8.057 | 1.16× |
| 640×4096×1024 | 48.761 | 41.839 | 1.17× |
| 640×1024×4096 | 75.698 | 75.828 | 1.00× |

The six larger eligible shapes improve by roughly 12–14% in elapsed time; the four other tested shapes remain close to unchanged. Default AMD/NVIDIA cooperative-matrix execution already groups weight loads and does not use this scalar fallback shader.

## Validation and limits

- CPU, CUDA, NVIDIA Vulkan and AMD Vulkan operator contracts passed, including 1,354,743 finite conversion comparisons per backend.
- CUDA narrow-math tests passed 6,205,420 comparisons with no differences, including all 256 legacy FP8 encodings.
- No model DLLs, SDKs, game files or frame dumps are included in the data. Games and installed configurations were untouched.
- The known-broken fused Vulkan feed-forward optimization remains disabled by default.
- Graph-owned weight, scratch and total backend allocation sizes matched before/after in every whole-model case. This is an ownership check, not a measurement of driver-private caches or system-wide RAM.
- Generated Metal arithmetic passed host simulation, including 10,000 FP8 oracle cases. This is distinct from an Apple compiler/device validation. HIP and SYCL device compilation/execution remain unverified on this machine.

## Reproduce

Use independently compiled, model-free control/candidate runtimes with an external model DLL. The maintained runners are `tools/test_cpu_operand_reuse.py`, `tools/test_frame_conditioning_reuse.py`, `tools/benchmark_portable_reuse.py` and `tools/benchmark_reuse_matrices.py`. CPU loaders and the frame header before this study are available in base commit `a151512`; pass a read-only copy of those files to the differential runners. Use `--scalar-matrices` for the forced Vulkan fallback comparison. Device UUIDs are discovered locally and omitted from this public report.
