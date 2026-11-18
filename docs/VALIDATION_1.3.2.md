# Phoenix 1.3.2 validation

- The Windows release build checks routing, capture queues, pixel conversion, WOW64 packet handling, addon loading, and embedded runtime extraction. Linux CI builds and checks the Vulkan layer and SDK package.
- The Windows development candidate completed 241 of 246 timed fixture cases after retries and passed all three DOS checks. Five dgVoodoo timing cases remained open. An RTX 5080 rerun passed all eight plain DX9–DX11 fixtures. DX9 x86 still showed a Phoenix throughput gap in that setup.
- The installed-game sweep produced capture pairs for 74 of 75 replays. The pairs were pending independent visual review at the end of the sweep. Oblivion Remastered exited before the input replay could attach.
