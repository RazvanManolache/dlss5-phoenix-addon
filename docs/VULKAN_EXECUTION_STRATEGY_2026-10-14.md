# Vulkan performance investigation: execution strategy

RTX 5080 16 GiB, driver 616.64; Ryzen 9950X3D, 96 GiB RAM.

The accepted local implementation is commit `07398bb`. Its paired 1080p
RGB32F temporal fixture measured 68.371 ms for model GPU submissions and
117.675 ms for the complete call, compared with 89.042 and 139.774 ms at the
previous checkpoint. These are still far from the requested 10 ms target.
The GPU-only native/NVIDIA fixture has a different frame/motion boundary;
its approximately 3.8 ms is not a complete-call comparison.

## What this pass actually tested

Each full-frame comparison below alternated three pairs of fresh processes,
twelve temporal frames per process. The first two frames were discarded for
timing; every one of the 72 outputs matched the saved reference exactly.
These paired baselines are contemporaneous and should not be mixed across rows.

| Experimental change | Baseline GPU / complete call | Candidate GPU / complete call | Conclusion |
| --- | --- | --- | --- |
| Native FP8 publication conversion | 67.857 / 114.762 ms | 67.244 / 116.230 ms | About 0.9% GPU gain; no complete-call win established. |
| Convert reusable matrix inputs to packed FP8 bytes once | 67.932 / 112.260 ms | 67.238 / 115.591 ms | Large matrix pieces improve, but only about 0.7 ms across the model. |
| Compile matrix, pointwise and row shapes into pipeline constants | 67.891 / 113.506 ms | 66.731 / 116.110 ms | About 1.7% GPU gain, substantial pipeline cost; retained as an experiment. |

The packed-input prototype adds a provider-owned 4 MiB device buffer.
The old caller report counts only adapter-requested tensors, so its unchanged
487,938,728-byte tensor field does **not** include this new buffer. The total
requested provider buffers become 492,133,032 bytes, before allocation rounding,
staging and driver storage. No memory saving is claimed.

The shape experiment retained the same tensor buffers but worker private commit
was approximately 811.88 versus 740.02 MiB, working set 327.12 versus 258.41 MiB.
Its first uncached frame took 37.8 seconds, versus about 0.5 seconds for the
baseline; later fresh-process runs benefited from the driver's persistent
compiler cache. This is not suitable for unconditional production enablement.

Large isolated packed-input matrix timings included all work in the selected
operation, including its packing dispatch:

| Real-input matrix M x N x K | Existing ms | Packed-input ms |
| --- | ---: | ---: |
| 640 x 4096 x 1024 | 0.084592 | 0.064864 |
| 640 x 1024 x 4096 | 0.099649 | 0.081586 |
| 2560 x 1536 x 512 | 0.068059 | 0.057675 |

The lesson is to rank candidates by their *weighted full-frame contribution*.
A 20-30% piece improvement can still affect too little of the frame to matter.

Other experiments:

- Leader-lane normalization matched output but was slightly slower.
- A nine-operation whole-window fusion matched its live output but took
  0.104029 ms versus 0.078187 ms. The compiled shader used 187 registers/thread
  and 45,056 shared bytes. Lower dispatch count alone did not make it faster.
- Streaming contraction weights with aliased shared storage matched output but
  took 0.108733 ms. A seven-operation feed-forward/QKV fusion also matched but
  took 0.088631 ms for the nine-operation comparison span.
- A 256-thread whole-window variant changed output and was rejected.
- Larger packed-input tiles were tested individually; the existing 64x64,
  128-thread choice beat the tested larger variants for the representative
  wide shapes. A 128x128, 256-thread version was markedly slower.
- Cooperative vector decode reduced some wide pieces but regressed narrow
  matrices. It is not enabled in production.
- Direct half casts and an observable pack/unpack alternative failed the
  projected-QKV equivalence check. They are rejected.
- Shape specialization helped the narrow 2160x64x512 piece by about 29% in time,
  but hurt another shape. Selective routing is essential.

## Native arithmetic is a different contract

`src/native/kernels/production/common/feed_forward_math.cuh` uses packed FP16
pair arithmetic and `mma.sync...f16.e4m3.e4m3.f16`. The portable Vulkan matrices
use FP32 accumulation. Native normalization also contains approximate reciprocal
and inverse-square-root operations; the portable reference uses explicit
refinement. These facts do not prove every native stage is equivalent to one
uniform FP16 policy.

An isolated prototype changed the four new cooperative shaders to FP16
accumulation. Representative pieces improved only roughly 1-6% and their output
hashes changed. It is retained as a **different numerical experiment**, not an
accepted optimization or verified native-equivalent implementation. Precision
alone did not explain the performance gap.

## A new NVIDIA execution route was verified at the buffer/kernel level

The installed driver advertises `VK_NV_cuda_kernel_launch`; its feature query
returns true and compute capability 12.0. A standalone prototype:

1. Creates the device, memory and command buffer through Vulkan.
2. Compiles our CUDA source to PTX and creates Vulkan CUDA module/function objects.
3. Passes Vulkan buffer device addresses in the CUDA launch packet.
4. Dispatches through `vkCmdCudaLaunchKernelNV` and synchronizes through Vulkan.
5. Reads the Vulkan-owned allocation and compares its contents on the CPU.

The elementary buffer probe passed 4,096 values exactly. More significantly,
two unchanged production native layout kernels, `phoenix_image_to_tokens` and
`phoenix_tokens_to_image`, passed 16,384 padded token-word checks, 12,288 reverse
checks and output guard checks. The 12x4 feature grid exercises real reordering
and zero-filled token padding. These are layout tests, **not neural inference,
image interop, full model parity or a performance measurement**.

This permits investigation of an explicitly named NVIDIA native-kernel Vulkan
route. It should reuse the optimized native launch plan, layouts and kernels,
while retaining the portable shader engine for other devices. It must not be
presented as an AMD/Android acceleration result or silently relabel a CUDA API
execution as Vulkan. The prototype actually uses the Vulkan launch extension.

Next implementation gates:

1. Validate sampled/storage image handles against native kernel texture/surface
   expectations. Buffer success does not establish image compatibility.
2. Add device-scoped module/function/buffer ownership and capability checks,
   including initialization failure, reset, graph destruction and fallback.
3. Run an actual native neural kernel on captured operands, including its
   scratch, gains and numerical output; then the full native graph.
4. Measure GPU-only and full-frame paths on identical inputs, controls, precision,
   history and padded extents. Validate temporal output, odd sizes and memory.
5. Keep the shared-memory/FP8/layout lessons available to the portable engine;
   the NVIDIA-only extension does not solve its cross-vendor performance.

Extension reference:
https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_cuda_kernel_launch.html

## Profiling and preservation limits

Nsight Systems hardware counters are blocked by the driver's profiling
permissions (`ERR_NVGPUCTRPERM`). An API trace attempt ran the benchmark but
reported no Vulkan events, so it provides no usable GPU timeline evidence.
The pipeline executable-statistics interface returned register/shared-memory
statistics but no internal code representations. No unsupported claim about
measured occupancy, cache misses, stalls or spills follows from those results.

The ignored workbench is `build/vulkan_architecture_20261014`. It contains the
candidate sources, recipes and fresh reports, including `strategy_full_summary.json`,
`*_strategies_result.json`, `*_full_results.json`, and both `cuda_launch_*probe`
folders. Each CUDA launch proof has a receipt binding its source and artifact
hashes. Alternatives were preserved with content deduplication in an external
local experiment archive; these private receipts are not release contents.

No experimental route from this pass has replaced the qualified addon packages.
No NVIDIA model DLL or extracted model is included in the preservation process.
