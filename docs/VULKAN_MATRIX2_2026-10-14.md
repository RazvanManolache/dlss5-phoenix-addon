# Vulkan workgroup matrices and fused blocks

Tested on RTX 5080 16 GiB, driver 616.64, Ryzen 9950X3D and 96 GiB RAM.
The final comparison uses three alternating pairs of fresh processes, each
running the same twelve 1920×1080 temporal RGB32F fixtures with fractional motion.
The first two frames are discarded for timing. All 72 frame outputs match the
saved reference exactly. Clocks are stock; this is a desktop measurement.

| Measurement | Previous cooperative/QKV/noise checkpoint | Accepted workgroup/fusion/shape path |
| --- | ---: | ---: |
| Median model GPU time | 89.042 ms | 68.371 ms |
| Median complete inference call | 139.774 ms | 117.675 ms |
| Owned device tensor buffers | 465.33 MiB | 465.33 MiB |
| Packed model weights | 144.51 MiB | 144.51 MiB |
| Scratch arena | 320.83 MiB | 320.83 MiB |
| Sampled host temporary capacity | 130.75 MiB | 130.75 MiB |
| Median worker working set | 258.70 MiB | 258.58 MiB |
| Median worker private commit | 756.73 MiB | 740.78 MiB |

GPU time improves by 23.2%, complete-call time by 15.8%. The 10 ms target has
**not** been reached. GPU intervals cover model submissions; complete calls
include CPU history preparation/resolve, uploads/readback and caller transport.
The native RTX/NVIDIA DLL GPU-only fixture uses a different image format and
motion boundary, so its timing is not a like-for-like complete-call comparison.

Device tensor bytes exclude driver/pipeline storage and staging. Host capacities
are sampled vector capacities, not total RAM. Process working set/private commit
also include the driver and runtime; their variation does not establish a RAM
saving. The code retains buffer-ID compatibility markers, without another host
weight copy. Additional pipelines can change driver memory and startup cost.

## Accepted changes

- Workgroup FP8 matrices load the original byte-packed weights directly through
  tensor addressing. Activations already proven to be on the finite FP8 grid
  convert from half storage, without a persistent expanded-weight workspace.
- A 32→128→32 feed-forward pair carries the activated hidden matrix between
  cooperative operations. FP16 contraction preserves the original accumulation
  behavior; isolated real-input time is 0.0140 ms versus 0.0344 ms.
- Single-head 32-channel QKV projection and preparation share a dispatch and
  shared projection tile. Q/K keep their ordered norm tree, V its half/FP8
  publication. Isolated time is 0.0157 ms versus 0.0217 ms.
- Narrow matrices use 32-row tiles. Matrices with at least 512 rows and 128
  output columns use 64-row tiles. Separate pieces and combined temporal runs
  validated this selection; selecting one size everywhere was slower.
- Exact extension/feature/shape queries gate these engines. Unsupported GPUs,
  misaligned layouts and partial fusion groups retain established paths.
  Legacy weight bytes 0x7f/0xff mean ±480 in the existing decoder and NaN in
  hardware FP8. Upload compatibility checks exclude those buffers from the new
  engine, invalidate markers after partial uploads and rebuild cached routing.

## Rejected or bounded experiments

The initial full-K FP8 feed-forward contraction matched real model fixtures but
differed by one ULP in a wider-range synthetic test. FP16 contraction fixed that
test while retaining most of the gain. Tiny tensor-store fusion shapes caused
device loss in synthetic tests; the accepted matcher requires complete 32-row
groups. Partial groups use individually bounded operations.

Whole-workgroup attention was slower (0.0490 versus 0.0363 ms for the tested
piece). Several subgroup/wider attention variants gave small isolated gains,
but were not promoted. Bigger 128-row/128-column matrix tiles also lost on the
tested pieces. These alternatives remain available for further experiments.

Validation includes 42 saved frames across CPU/CUDA/HIP/SYCL/NVIDIA Vulkan/AMD
Vulkan, 80 additional accelerated/scalar size/seed/reset/multipass frames,
72 synthetic fusion/legacy-encoding/tail cases, strided-batch 32/64-row matrix
comparisons with row/column tails, and the existing NVIDIA/AMD operator suites.
The addon core passed fifteen CPU suites and two software D3D12 suites. This
does not certify every GPU, replacement model or input distribution.

The workbench is `build/vulkan_architecture_20261014`; selected production files
are in `adaptive_runtime`. Source variants, our binaries and evidence are also
preserved in an external local experiment archive, with
content deduplication. Vendor runtimes and model DLLs remain external inputs.

All four switches default on when supported: `vulkan_workgroup_fp8`,
`vulkan_workgroup_shape_tiles`, `vulkan_fused_feed_forward`, `vulkan_projected_qkv`.
Each can be disabled independently. Detailed dispatch profiling retains separate
fusion operations so attribution remains meaningful.
