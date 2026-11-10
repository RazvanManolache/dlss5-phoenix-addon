# Phoenix DLSS5 1.1.0 validation

The completed validation campaign recorded **2,219 passed case/check units**.
The cumulative game-compatibility campaign covers **76 games**.
Case/check units include contract checks, model/backend cases and profile
checks; they are not 2,219 distinct games or test executables.

- Both Original and Lightweight FP8 were exercised across the available Windows processing routes.
- Eight live fixture/model checks passed with both add-ons loaded and resource hashes verified.
- Six qualified D3D12 transport routes and 48 fresh helper-process repetitions passed.
- Twenty-two longer-sampling profiles passed. Native pipeline feature profiles and NVIDIA comparisons cover temporal/history behavior, guides, masks and appearance controls where supported.
- Current source, runtime and package inventories were checked against their build receipts before release preparation.
- Release packaging additionally checks architecture, DLL loading, embedded extraction, source closure, SHA-256 inventories and absence of model data and private host identifiers.
- Linux release builds run portable contract tests and relocatable-package checks in GitHub Actions. Those checks establish build/package coverage without claiming GPU execution on the hosted runner.

## Known limits

Four Vulkan 5120×2880 cases fail at the preferred default batch size with an
operand-offset error; the same cases pass with an explicit batch size of 512.
Four Vulkan 7680×4320 cases exceed available resident GPU memory. Two optional
direct x86 driver-host cases retain an initialization/frame validation issue.
These ten failures are retained findings, separate from the passed count.

Linux native CUDA currently uses a bounded reset-only SDR path; Windows native
CUDA has the extended temporal and frame-services integration. Native CUDA
kernels shipped here target SM120. Other RTX architectures retain their
separate support and qualification boundaries.

Android benchmarking was unavailable. Apple Metal, usable current HIP/SYCL and
physical older RTX hardware were outside this campaign's available devices.
See [execution mechanisms](EXECUTION_MECHANISMS.md),
[Windows CUDA](WINDOWS_CUDA.md) and [Linux native CUDA](LINUX_NATIVE_CUDA.md).
