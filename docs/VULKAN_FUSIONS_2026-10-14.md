# Vulkan fusion experiments — 14 October 2026

Three approaches were implemented and tested independently, then in combinations: shared-memory feed-forward fusion, combined Q/K/V preparation, and GPU frame preparation/resolve. Production keeps combined local Q/K/V preparation and seeded GPU noise caching. Feed-forward fusion and GPU resolve remain local alternatives.

The final production-source comparison used four balanced fresh processes per variant, the same 12-frame 1080p fractional-motion sequence, and two discarded warmup frames. All 144 outputs matched the saved reference exactly.

| 1080p measurement | Previous cooperative checkpoint | QKV only | QKV + noise cache |
| --- | ---: | ---: | ---: |
| Median model GPU time | 92.137 ms | 90.295 ms | 91.194 ms |
| Median complete inference call | 141.760 ms | 140.136 ms | 138.793 ms |
| Owned device storage | 453.38 MiB | 453.38 MiB | 465.33 MiB |
| Scratch arena | 308.88 MiB | 308.88 MiB | 320.83 MiB |
| Retained CPU noise vector | 11.95 MiB | 11.95 MiB | 0 MiB |

The combined production path reduced complete-call time by 2.09% in this comparison. QKV alone gave the better GPU-only median; noise caching gave the better complete-call median. Stock clocks and shared desktop use limit the precision of these small whole-frame differences. The device noise allocation replaces the retained CPU noise vector rather than adding a second copy. This is a storage-capacity accounting result, not a claim about total process or driver RAM. Model weights are unchanged at 151,526,056 bytes.

Isolated real-input QKV groups improved by 14–33% for the tested head counts. Eight lanes retain the original normalization tree for each head, while V keeps its half-then-FP8 publication. The new route only matches complete local groups and does not cross command-buffer submission boundaries. Global and split-head attention, other vendors, and detailed per-dispatch profiling retain their previous scheduling.

Four feed-forward candidates were tested: one, two and four subgroups sharing rows, plus four subgroups splitting hidden columns. All matched the contracted output, but the isolated 1080p pair took 0.0399–0.0543 ms versus 0.0343 ms for the existing separate pair. Combining feed-forward fusion with the otherwise winning changes also worsened full-frame timing. None was enabled in production.

GPU resolve avoided head readback and tested both transient scratch reuse and retained preparation planes. Ordinary shader arithmetic differed in 162,069 of 6,220,800 output components in the first 1080p temporal frame, with maximum absolute error 5.96e-8. A more accurate exponential/reciprocal implementation reduced this to 18 components, but still failed exact matching. It passed a smaller moving, multipass test; that did not establish 1080p parity. Retaining all preparation planes also added 61.76 MiB of device storage. Both versions were rejected.

Validation of the accepted code includes:

- Exact separate-versus-fused Q/K/V comparisons for 1, 2, 4 and 8 heads, with 1, 17, 63, 64 and 129 rows.
- 80 additional exact output comparisons across accelerated and scalar Vulkan at 1×1, 96×64, 321×193, 513×257 and 1920×1080. These change seeds (including zero and UINT32_MAX), reset history, exercise maximum supported intensity, and use multiple passes.
- The portable saved regression suite and NVIDIA DLSS5/AMD backend operator contracts, including 1,354,743 finite conversion checks per device.

Library configuration can disable either accepted optimization independently:

```json
{"vulkan_fused_qkv": false, "vulkan_cache_noise": false}
```

The defaults apply to the tested NVIDIA Vulkan route. Noise caching regenerates and uploads on a seed change or graph rebuild, releases the CPU vector after successful inference, and invalidates stale cache state before an upload can fail. Other backends retain their existing policy.

Hardware: RTX 5080 16 GiB, NVIDIA driver 616.64, Ryzen 9 9950X3D (16 cores / 32 threads), 96 GiB RAM. AMD checks use the integrated gfx1036 GPU, not a discrete Radeon card. Tests are synthetic library/runner tests; no new game-rendering or physical Android/Intel-GPU claim is made.

Historical local experiment receipts used `build/vulkan_fusions_20261014`. Frozen sources, recipes and selected alternative binaries were also preserved in an external local archive of about 3.5 MB, without a model or NVIDIA DLSS5 DLL. The accepted runtime was `build/vulkan_fusions_20261014/final_runtime`. These paths describe the historical experiment, not release contents.
