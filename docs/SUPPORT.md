# Current support and validation

This describes 0.9 and its two-model and processing updates. It is not the
contents of every older published package.
[Current benchmark](benchmarks/current/README.md) separates device measurements
from source support. External NVIDIA model weights and vendor SDKs are supplied
separately.

## Models

| Property | Original | Lightweight FP8 |
| --- | --- | --- |
| Stable ID | `original` (default) | `small-fp8` |
| DLL resource | `WEIGHTS_HT` | `WEIGHTS_CC_SMALL_FP8` |
| NVIDIA reference preset | 1 | 2 |
| Learned blocks / decoded tensors / packed records | 71 / 658 / 153 | 31 / 302 / 77 |
| Full native neural launches | 154 | 78 |
| Eight controllable central transformers | Learned blocks 31–38 | Learned blocks 11–18 |
| Native fixed work + repetitions | 114 + 5 × retained repetition sum | 38 + 5 × retained repetition sum |
| Learned temporal blend | Block 70, 0.73974609375 | Block 30, 0.76708984375 |

Both resources are present in the authenticated 310.9.16 DLL. FP8 names the smaller
model/resource architecture; it does not mean every portable operator runs native
FP8 arithmetic. Explicit model selection never silently substitutes another model.
Changing models resets temporal history. NVIDIA preset selection and Phoenix
appearance style are separate controls.

## Execution backends

| Backend | Source/build scope | Both models | Fresh verification on this host |
| --- | --- | --- | --- |
| Native RTX | Windows x64, D3D12, authenticated source-built kernel bundle; current bundle targets sm_120 | Yes | RTX 5080, both Phoenix models and both NVIDIA presets, ten extents from 320×180 through 8K |
| CUDA | Optional Windows C++ provider; NVIDIA CUDA compiler/runtime | Yes | RTX 5080, five extents through 4K |
| Vulkan | C++ adapter and shader provider; capability-checked scalar/fused variants; Windows/Linux/Android build paths | Yes | RTX 5080 through 4K with ordered FFN; integrated AMD Radeon at 320×180 and 1080p |
| CPU | Portable C++ reference; optimized Windows provider with AVX and worker controls | Yes | Ryzen 9950X3D, 32 threads at 320×180/1080p and one thread at 320×180 |
| Metal | Experimental macOS Apple silicon, unified memory and Apple GPU Family 7 requirement | Yes, shared graph | Source/host graph contracts only; no Mac or Apple compiler available |
| HIP / AMD | Optional Windows HIP provider; matching AMD SDK/runtime and target architecture | Yes, shared graph | Source/graph contracts only in this refresh; no local SDK/runtime |
| SYCL | Optional Windows provider; compatible SYCL compiler and CPU/GPU runtime | Yes, shared graph | Source/graph contracts only in this refresh; no local SDK/runtime |

The experimental fused feed-forward optimization previously failed 4K reset
repeatability in both models on RTX Vulkan. The current unreleased runtime defaults
`vulkan_fused_feed_forward` to `false`, retaining the full graph through ordered
feed-forward pairs. Fresh broad measurements exercise that safe default without a
caller override. Explicit opt-in remains experimental: the optimized shader/driver
defect is not repaired. [Targeted review validation](benchmarks/current/review-fixes.md)
also checks repeated 4K resets; historical failure evidence is retained separately.

CMake currently exposes CPU, Metal and optional Vulkan builds, plus the Windows
x64 native extension. Optional CUDA/HIP/SYCL providers use the Windows build tools;
this table does not claim corresponding Linux CMake targets. Platform build paths
are not current device tests on Linux, Android or macOS. No discrete AMD or Intel
GPU was measured. Earlier Original-only HIP/iGPU and SYCL/CPU reports are historical.

## Workload, appearance and frame controls

Both Phoenix models support a 1–8 transformer depth ceiling, 0–4 repetitions per
central transformer and 1–10 whole-network passes in the addon/shared portable
runtime. Passes retain separate temporal histories. Native passes share immutable
weights and serial workspace. Reducing depth or changing repetitions is an
approximation that changes the trained execution.

GPU-budget feedback uses measured GPU intervals for the selected model/backend;
CPU timings are not substituted. Minimum depth and fixed encoder/decoder work can
prevent reaching a requested budget. Application-FPS feedback needs real active
application presentation timestamps. The direct addon owns that cadence; portable
library callers must supply it. Final-frame/helper panels expose GPU budgets and
keep game-FPS targeting unavailable. Offscreen benchmark iteration rate is not game
FPS. [Scheduling details](PORTABLE_MODEL_CONTROLS.md).

The standalone `phoenix_native_*` C API remains single-pass with repeat controls;
the addon/direct runner and portable C API own the multi-pass scheduling described
above. The experimental in-process x86 native driver-host route remains
Original-only at the full eight-transformer, one-repetition schedule. The separate
x64 helper/library can execute either model.

Appearance support includes tone, structure, skin, intensity, Default/Natural/
Cinematic styles, model-supplied history blend and sampling controls. Explicit
engine inputs enable per-surface RGB control maps, protected premultiplied UI and
forward/inverse distortion. These recovered/experimental semantics are not a claim
of NVIDIA-equivalent advanced options or automatic object/UI recognition.
[Appearance guide](NEURAL_CONTROLS.md).

The processing pipeline adds Off/Render/Upscaled/Present insertion, scaling and
aspect controls, input encoding/gamut/transfer conversion, NR signal selection,
exposure/diffuse white, area/bilinear/nearest downsampling, bilinear/bicubic/nearest
upsampling, RGB residual reconstruction, UI policy and guide reuse. Availability
depends on the frontend and actual frame guides. Color-only transport cannot
manufacture temporal guides. [Pipeline and exact bounds](PROCESSING_PIPELINE.md).

## Integration and distribution boundaries

ReShade frontends cover D3D9/10/11/12, OpenGL and Vulkan routing on 64-bit Windows,
with x86 and x64 addons. A capture API is separate from an inference backend:
Vulkan capture does not imply Vulkan compute, and an x86 game may use an x64
helper. Direct game DLSS interception, final-frame capture and external feeders
have different resource/guide contracts. See [graphics compatibility](GRAPHICS_COMPATIBILITY.md)
for the actual per-route evidence; the current neural benchmark does not revalidate
every graphics API or launch a game.

Built-in spatial reconstruction, DLSS Super Resolution and XeSS integration have
their own runtime/guide requirements and are not benchmarked by the isolated NR
tables. Comparison/split/difference views, presets, model loading/status, retained
timing rows and optional device offload are frontend features. Dual-discrete-GPU
gaming performance and VR support remain unverified.

Current development binaries under `artifacts/current/processing` are unembedded
builds. Published 0.82 packages predate these two-model and scheduling updates;
0.9 includes them. Release-package checks are recorded in its `VALIDATION.json`,
separately from the benchmark build identities.
Source support, compilation, CPU contract tests, isolated GPU inference, live-game
validation and public release availability are distinct checkpoints.
