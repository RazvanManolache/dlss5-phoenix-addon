# Make the pixels take your route

Phoenix DLSS5 can do more than one neural pass with a few sliders. The optional
processing chain lets you choose the order of neural rendering, reconstruction,
color treatment and multiscale fusion. Leave it off for the established path.
More stages mean more work, not automatically better pixels.

## Controls that reach the renderer

- Enable, disable and reorder the four stages in the add-on's pipeline panel.
- Adjust brightness, contrast, saturation and gamma in the color stage.
- Choose the existing spatial scaler, NVIDIA Image Scaling (NIS), or an installed
  tiny 2x checkpoint. VSR is deliberately unavailable, not a hidden working option.
- Blend two or three independently evaluated neural resolutions with experimental
  multiscale fusion. Each branch/pass has independent temporal history.
- Prefer engine motion vectors; optionally estimate NVIDIA optical flow when
  available, reject low-confidence history, and reset on scene discontinuities.
- Opt into pipeline monitoring and a coordinated age budget. Monitoring off avoids
  its timing queries. Reported source-to-Present API time excludes physical scanout.
- Use comparison zoom/loupe and optional screenshot settings/hash receipts.
  ReShade's saved-image event cannot prove an exact source-frame association.
- Keep last-working-settings recovery enabled to recover from failed trials.
  It is not crash isolation or a guarantee that a graphics driver can recover.

## Where it runs

| Route | Available scope |
| --- | --- |
| Local final-frame D3D12 bridge / native SDK | Shared stage graph, independent processing/output sizes, color, scaling, fusion and capability-gated frame services |
| x64 shared portable frame worker | Shared D3D12 stages around separate native portable neural workers; independent histories and reported worker PIDs; host transfers remain |
| Equal-extent portable C API | Color/fusion metadata with equal processing/output extents; fixed pipeline priority; not all shared D3D12 services |
| Experimental x86 native library/driver bridges | Existing neural controls; custom chain, flow, scene detection and pipeline monitoring are explicitly rejected |
| Other insertion hooks | Existing supplied guides/reset flags and route-specific controls; final-frame services are not assumed |

Native stages use D3D12. A portable neural backend does not make every stage
portable to every platform. NIS supports the documented 1x-2x range; the fixed-size
tiny 2x stage needs a 50% processing canvas. Missing checkpoints and unsupported
dimensions/providers are rejected. Appearance controls require a route advertising
the shared frame-service capability.

## Costs and limits

[Retained 1080p pipeline measurements](benchmarks/current/pipeline/README.md)
include host upload, synchronization and readback. They are not interchangeable
with the main README's GPU-resident Phoenix DLSS5 / NVIDIA DLSS5 comparison.
In that fixture, two-scale fusion increased Original host time from 7.72 to
11.51 ms and Lightweight from 6.01 to 8.85 ms. Fusion softened facial detail.
The tiny scaler took about 58 ms and produced conspicuous artifacts. It remains
an experiment, not the recommended default or a trained replacement for DLSS5.

New model loading streams hashes and selected resources instead of allocating
the entire large DLL. Native uploads authenticate the bytes actually transferred
and use bounded fallback staging when GPU-upload allocation fails. x86 startup
also checks free/contiguous address space; the separate x64 helper remains the
compatibility escape route. These changes target startup pressure, not a claimed
reduction in steady-state model VRAM or faster inference.

The [implementation ledger](PIPELINE_IMPLEMENTATION.md) records prior small-fixture
checks separately from [current release validation](VALIDATION_0.13.md). No fresh
GPU or gameplay check is implied by a successful release build. Physical older
RTX cards, AMD-only D3D12 presentation and unavailable HIP/SYCL/Metal hardware
remain outside the measured matrix.

## SDK and companions

`phoenix-process` supports file jobs and PXF1 framed streams with source timestamps,
fixed offline quality and existing-output preservation. The portable `phoenix_frame`
tool also forwards supported chain metadata. See [library contracts](LIBRARY.md).
The companion player/Everything integrations live in separate repositories; this
add-on release does not include their applications, media helpers or models.

Thanks to the [optical-flow, NIS and tiny-model upstreams](UPSTREAM_CREDITS.md).
Their source licenses/notices are retained; model weights remain external.
