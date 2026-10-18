# Vulkan cooperative-matrix checkpoint — 14 October 2026

The RTX 5080 path now stages aligned packed-weight words in column-major shared memory, reuses each 32-element reduction stage for two ordered matrix operations, and keeps each attention subgroup's score normalization and context scatter within its own rows. FP8 weights still decode to FP16 operands; accumulation and publication order are unchanged.

Four balanced fresh processes per variant ran the same 12-frame 1080p fractional-motion sequence, discarding the first two frames for timing. Every output matched the saved baseline exactly.

| Measurement | Previous shaders | New shaders |
| --- | ---: | ---: |
| Median model GPU time | 98.080 ms | 91.610 ms |
| Median complete inference call | 160.676 ms | 153.897 ms |
| Owned device allocations | 453.38 MiB | 453.38 MiB |
| Scratch arena | 308.88 MiB | 308.88 MiB |

This is a 6.60% GPU-time reduction and a 4.22% complete-call reduction. These portable-runner calls include host preparation and resolve; they are not directly comparable to native Phoenix DLSS5's GPU-only timings. Stock clocks and shared desktop use remain a measurement limitation. Worker RAM was essentially unchanged (about 270 MiB working set and 736 MiB private bytes).

Validation includes 96 exact paired temporal outputs, 42 saved regression frames across all six runner routes, and 40 additional exact edge/control/multipass outputs in both accelerated and scalar Vulkan modes. The edge extents include 1×1, 96×64, 321×193, 513×257 and 1920×1080. Both addon architectures built successfully with this runtime; this checkpoint has not been published or installed into a game.

A larger FP8 cooperative-matrix prototype remained about three times slower than the accepted FP16 path in the selected dense operators and was rejected. A workgroup-tail prototype failed intermediate output hashes and was corrected before promotion. A 64-element reduction stage lost time on common window shapes; the accepted stage is 32.

Local alternatives, exact-output receipts, timing runs and the accepted package are retained under `build/vulkan_round2_20261014` and `build/vulkan_round2_package_20261014`. The profiler now records hashes for each selected operator's written output regions, outside its timed interval, so future operator experiments can catch internal errors before final image publication.

Hardware: NVIDIA RTX 5080 16 GiB, driver 616.64; AMD Ryzen 9 9950X3D, 16 cores / 32 threads; 96 GiB system RAM. The additional AMD route uses the machine's integrated gfx1036 GPU, not a discrete Radeon card.
