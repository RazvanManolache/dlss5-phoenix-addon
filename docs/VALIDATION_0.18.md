# Phoenix DLSS5 0.18 validation

## Release gates

The standard and experimental builds run the maintained CPU contract suites and software D3D12 tests. Architecture-matched minimal ReShade hosts load the final addons, check the public feeder ABI and channel identity, balance registration and detach, and release their module references.

Targeted CPU probes use the real addons and MinHook with stub NGX/Streamline exports: temporary-device reload cycles, exactly-once forwarding, foreign resource handles, and switching between two independent ReShade providers. A stale provider's unload must not unregister the current provider. Fixture option tests cover combined shutdown/MSAA and Vulkan queue flags without creating graphics devices.

Optional Input checks load the final x86/x64 binaries against an existing offscreen, non-activating test window, without swapchain callbacks. They check window discovery and hook activation, mailbox status, stale-session rejection and balanced unload. Separate legacy-input tests exercise DirectInput 7/8 A/W, buffered polling and test-owned window/cursor handling without sending desktop input.

The receipt binds final binary hashes, source inputs and actual test outcomes. Embedded payloads are recursively audited, including nested addons. Runtime companions and native kernels are checked against the preceding published package; unchanged binaries are not described as newly benchmarked.

Source and binary packages are scanned for excluded model data, NVIDIA DLLs, private paths, credentials and diagnostic artifacts. The source ZIP equals the audited committed tree. Public downloads are retrieved independently and SHA-256 checked.

## Boundaries

No retail games, hardware-GPU inference, fresh benchmarks, physical older-RTX qualification or full graphics API matrix are run for this publication. Software-device and CPU evidence do not establish visual quality or end-to-end game compatibility. Previous benchmark results retain their original build and test scope.
