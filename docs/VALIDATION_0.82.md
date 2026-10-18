# 0.82 validation scope

The release combines source/build checks with full-model fixture evidence.
The downloadable `VALIDATION.json` records final binary, archive and source
identities and separates retained pre-release measurements from fresh checks.

Pre-release evidence includes 285 native correctness frames, exact repeated
resets at 11 sizes through 8K and fresh sessions at the two failing sizes;
native/portable artistic controls; exact CUDA/Vulkan fixture outputs; optional
plane snapshots and local/helper/x86-client replay; and actual shared-panel
interactions on both bitnesses. These claims apply to the named fixtures and
runtime identities, not automatically to every rebuilt binary.

Fresh release checks rebuild the scoped source, run the relevant contracts and
unit tests, verify native reset correctness, and inspect nested embedded
resources for the current graph, runtime and model-free declarations. Archive
inventories and public downloads are verified against SHA-256 checksums.

Fresh rebuilt-binary checks pass 33 Python tests; 15 Windows x64, 8 Windows
x86 and 8 Linux portable/native suites; 20 add-on CPU suites; and 2 software
D3D12 suites. Both panel bitnesses pass 1,387 widget assertions across 24
rendered headless panels each. Full-model artistic fixtures pass 9 cases each
on native RTX, CUDA and Vulkan. CUDA/Vulkan outputs are exact for this fixture;
native protection/composition errors stay within 0.001 (FP16). A separate CPU
fixture accepts three full-model frames at 96x64.

The rebuilt native resolution test accepts 209 full-model frames at 11 sizes.
The packaged runtime is checked separately after extraction from the actual
add-on. The [machine-readable receipt](validation/0.82.json) provides the final
counts, reset hashes, binary identities and embedding checks.

The 1024x768 failure and 4K reset corruption were caused by stale graph metadata
and short tiled buffers. See [the diagnosis](NATIVE_CORRECTNESS_0.82.md).
The native kernels and model weights were unchanged by that repair.

No retail game presentation, complete Streamline NR provider, arbitrary HDR UI,
curved moving-field/vendor equivalence, universal device support or NVIDIA-output
parity is established here. CPU-only/software D3D12 tests, headless widget
interactions and model inference remain distinct evidence categories. The
portable optional-effects CPU overhead and its earlier 8K allocation limits
remain open performance work.

Published source and archives exclude model DLLs/weights, local configurations,
hardware-specific device selections, private host paths, captures and raw logs.
Third-party notices accompany the binaries. Diagnostic prefixes are not full
network render acceptance. GPU tests require explicit invocation.
