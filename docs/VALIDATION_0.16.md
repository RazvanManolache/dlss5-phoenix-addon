# Phoenix DLSS5 0.16 validation

## Targeted paired-capture checks

- The x86/x64 WARP fixture copies one original texture, changes that same texture, then copies the modified pixels on the same command list. Both PNGs are decoded and compared with distinct expected RGBA values at a non-aligned 17×9 extent.
- A capture cannot complete from a recorded-but-unsubmitted list or an unretired fence. A discarded recording cancels its capture. Repeated captures get distinct directories, and readback bytes return to zero after completion.
- The staging-helper latch ignores an older result, retains only the next accepted input, handles rejected submission, and pairs only an equal frame token. Its exported PNGs are decoded too.
- Format normalization, truncated readback rejection, opaque screenshot alpha and invalid dimensions are checked. No model or hardware GPU is used.

## Shared UI

The real ImGui interaction fixture passes on x86 and x64: 1,869 widget checks, 1,603 dashboard checks and 40 policy/action checks per architecture. It covers all four tabs at 360/700/1,100 pixels, button dispatch exactly once, no dispatch while disabled, the seventh binding slot, duplicate/reserved keys, Save, scrolling and host-style restoration. Existing persisted rendering settings and shortcut indices are preserved.

## Release package checks

Standard and experimental-compatibility addon packages are rebuilt from current source. Each runs 24 CPU suites and 3 software D3D12 suites, including the new paired-screenshot test. Architecture-matched minimal ReShade hosts load and unload the actual final DLLs and check their public ABI. These hosts do not simulate gameplay.

Embedded runtime resources are extracted and hash checked on both architectures. Recursive audits verify the x86 package contains the new nested x64 addon and that each embedded graphics helper matches its standalone build. The tracked source tree is audited for compiled artifacts, models, private paths and credentials before source archiving. Published downloads are checked against their local SHA-256 inventories.

The optional Phoenix Input binaries retain their 0.15 source-bound smoke receipts and byte identities. Portable runtimes and native kernels also retain their verified 0.15 identities; earlier hardware qualification is not relabeled as a new result. No NVIDIA model DLL or model weights are distributed.

## Limits

No new retail-game, physical-GPU inference, broad graphics-API screenshot matrix or performance benchmark is claimed here. UI controls and capture state are wired for the maintained routes; the targeted fixtures establish software pixel/frame association, not every game's hook placement or final display color. PNG captures are SDR and opaque; they do not preserve HDR range or auxiliary alpha. Direct asynchronous game offload remains unavailable for exact paired capture.

`VALIDATION.json` binds the shipped binaries, source publication commit, retained resources and detailed test receipts.
