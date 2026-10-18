# Native tile-storage correctness repair

The source already described a corrected native layout, while a retained kernel
bundle still supplied an older graph. Rebuilding the native DLL alone left that
external graph in use. Its hash was valid, but its buffer layout was obsolete.

| Extent | Logical tokens | Required storage tokens | Result with old graph |
| --- | --- | --- | --- |
| 1024x768 | 240 | 256 | First-frame GPU hang |
| 3840x2160 | 2160 | 2176 | Reset output varies |

The kernels write complete 32-token tiles. Reserving only the logical token
count left 1024-channel features 16 KiB short and 2048-byte reduction scratch
32 KiB short at both sizes. Several tails overlapped another live tensor.

At 1024x768, launches 1-96 completed, while launch 97
(`cc_vit_1d_projection_wait_fp8`) triggered `DXGI_ERROR_DEVICE_HUNG`, HRESULT
`0x887A0006`. Its final scratch reserved 491,520 bytes but needed 524,288;
the required end lay beyond the entire workspace allocation.

At 4K, encoder/repack input was repeatable, but the first observed difference
appeared in transformer feed-forward contraction and propagated through later
stages. Extra barriers and isolated kernel experiments did not repair the old
layout. The corrected graph made reset output and the captured intermediate
tensors exact using the same 41 kernel binaries and unchanged model weights.

The newer graph also reserves the rounded downsample zero border throughout its
lifetime and corrects spatial/repack launch coverage. This prevents that border
from being reused as another tensor's storage, addressing the earlier
intermittent 3840x1600 fixture as well.

The loader supports an additive, separately pinned `native_graph_v2.json` while
retaining the legacy graph for unchanged SDK readers. Fresh source-built
bundles can use the corrected layout as `native_graph.json`. The runtime rejects
obsolete token/repack/padding layouts, the packager checks the selected graph
against source, and embedding retains every graph named by the manifest.

Pre-release checks covered 11 sizes through 8K (209 full-model frames), plus
four fresh packaged sessions at 1024x768/4K (76 frames). Reset RGB maximum error
was zero in every checked case. Nine native advanced-feature fixtures passed
their FP16 bounds, along with 15 runtime suites and 31 Python tests at that
checkpoint. Fresh release-build checks are recorded separately in
[validation details](VALIDATION_0.82.md).

These are structured SDR fixtures, not arbitrary game, HDR or NVIDIA-output
parity acceptance. No post-fix performance sweep is implied.
