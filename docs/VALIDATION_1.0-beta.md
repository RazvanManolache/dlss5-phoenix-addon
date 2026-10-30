# Phoenix DLSS5 1.0 Beta validation

The authoritative per-file and test results are in the release's `VALIDATION.json` and `SHA256SUMS.txt`. Source support, compiled device images, isolated inference and retail-game validation have different scopes.

## Fresh release checks

- Rebuild matching x86/x64 addons and their graphics helper from the published source.
- Both package cores passed 25 CPU suites and four software D3D12 suites, including shared-panel, capture and processing contracts. Final package checks also cover actual DLL load/unload and hook lifetime.
- Recursively decompress and inspect embedded payloads. Bind every runtime/kernel byte to the frozen, independently inventoried package inputs; verify the nested x64 core and graphics helper against their build receipts.
- Rebuild the portable host, CPU and Vulkan components for their packaged architectures, the x64 CUDA backend and the native library. Exercise both model families with bounded CPU fixtures; keep GPU execution evidence separate.
- Build and smoke-check the optional Input companion without sending input macros.
- Audit the staged source and all ZIP contents for excluded models, logs, captures, private paths and secrets. Verify source ZIP blobs against the commit and all published assets against their checksums.

These are the required gates for publication. Their completed results, counts and hashes are recorded in the machine-readable release receipt rather than inferred from compilation alone.

## Earlier GPU evidence retained by identity

- The selected native kernel bundle contains 410 compiled native/style device images across 41 shared modules. Older physical GPU generations remain unvalidated.
- The matched native Phoenix DLSS5 / NVIDIA DLSS5 study checks both models through 8K on RTX 5080. The instruction-style and multiresolution studies retain their own fixtures and matching-control identities.
- The portable reuse study records exact-output CPU and Vulkan before/after comparisons. Its AMD measurements are on integrated Radeon graphics, not discrete AMD cards.
- The resolution investigation uses 112 isolated processes to distinguish pixel growth, padding/token boundaries and register/occupancy tradeoffs. Rejected register budgets are not shipped as production optimizations.

Repackaging or recompiling a host DLL does not turn these studies into fresh GPU checks of that new DLL. No new retail-game matrix, physical older-RTX inference, Metal device test, HIP device test or SYCL device test is claimed for this release procedure.

[Native results](benchmarks/current/README.md) · [Instruction tuning](NATIVE_TUNING_RESOLUTIONS.md) · [Portable reuse](PORTABLE_REUSE_STUDY.md) · [Scaling profile](NATIVE_SCALING_PROFILE.md) · [Support matrix](SUPPORT.md)
