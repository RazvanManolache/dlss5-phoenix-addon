# Standalone port validation — 14 October 2026

This is a development/compatibility checkpoint, separate from the native RTX
performance charts. No Mac or Android GPU performance is measured.

- Windows x64, MSVC 19.51: C API, platform/model-parser and CPU operator suites pass.
- Ubuntu 22.04 under WSL, GCC 11.4: shared libraries and the same three suites pass.
- A full-model **64×48** synthetic RGB frame, all eight global blocks enabled,
  one pass, 16 CPU workers, completed on both hosts. The portable graph internally
  pads this to **336×320**, so this is not a low-resolution speed benchmark.
  Both outputs were **byte-for-byte identical**, 9,216 float values. The outputs
  were finite; all **1,673** recorded graph commands executed.
- Both reported **151,526,056 bytes** of resident weights and **38,551,552 bytes**
  of scratch (**190,077,608 total**). This excludes host frame buffers, callers,
  model loading peaks, libraries and driver allocations; it is not process RAM.
- Vulkan source/shader generation was factored out of the Windows compiler path.
  The generated desktop C provider and all **32 SPIR-V modules** match the
  existing tested provider byte-for-byte. Android additionally resolves the
  buffer-device-address function dynamically for the API 28 loader boundary;
  the arithmetic shaders are unchanged.
- Newly built Windows Vulkan backend operator tests pass on **RTX 5080**:
  matrix tails/layouts/strides, fused residuals, attention/head transforms,
  local/global row normalization, frame packing and **1,354,743** finite conversion
  comparisons. This validates reuse on desktop, not on a phone GPU.
- Installed SDK frame clients also ran the same full-model fixture on Windows
  CPU, Windows Vulkan and Linux CPU. All three outputs matched byte-for-byte:
  SHA-256 `9b1f084f227995f363d4df12eebc18fb197c2cb92277eb2b48ecc3805db59939`.
  Vulkan executes **1,317** commands with its supported fusions; CPU executes
  **1,673**. Resident weight/scratch totals agree for this fixture.
- Android ARM64, NDK **27.3.13750724**, Clang 18.0.4, API 28: CPU/runtime/client
  and Vulkan provider/backend cross-compile. This is not an APK or device run.
- Linux Vulkan provider/backend build against the Linux loader. WSL's available
  Vulkan devices are not equivalent to testing a native Linux desktop GPU.
- Generated Metal arithmetic passes host checks over all non-NaN binary16 inputs
  and eight publication modes. This checks the source adaptation only; it does
  **not** execute Metal. Separately, macOS 14 ARM64 GitHub CI successfully compiles
  both the Objective-C++ backend and MSL with Apple's compiler, and passes the
  C API/platform/CPU operator suites. A device suite is provided; CI only executes
  it if enumeration finds a supported GPU. This run exposed no supported Metal
  GPU, so device tests were explicitly **not executed**. No full-model Metal
  result is certified.

[All five CI jobs passed](https://github.com/RazvanManolache/dlss5-phoenix-addon/actions/runs/38491922392)
for binary checkpoint `a78569df208b5eabfc2d0b70032c519b4055be35`: Windows,
Linux, macOS, Android ARM64, and a separate Linux/Android Vulkan build. Vulkan's
host compiler was built from pinned glslang 16.6.0 and Vulkan-Headers 1.4.365.
Local release builds were made from a clean tracked-source archive, without
other in-progress changes in the checkout. SDK packages contain neither model
data nor NVIDIA's model DLL.

No model files, DLLs or private frame jobs are included in the published source.
The supplied NVIDIA DLSS5 DLL is authenticated/read as PE resource data by the portable
runner. Its executable code is not loaded by the standalone inference path.

## Remaining acceptance work

1. Build on macOS with Xcode, run `metal_operators` on Apple silicon, then run
   full-model and temporal comparisons against the existing reference. Measure
   Metal compilation time, warm execution, memory and sustained performance.
2. Run Android operator/full-model/temporal tests on actual Adreno/Mali devices
   that meet the Vulkan capability requirements. Measure thermal behavior and
   memory budgets; there is no playable-frame-rate claim yet.
3. Run Vulkan on native Linux desktop GPUs with validation layers. WSL CPU tests
   and cross-compilation do not establish native GPU parity or gaming support.
4. Add application-specific texture integration and performance-oriented Metal
   matrix kernels. The existing host-plane API intentionally supplies a first
   integration boundary, not a zero-copy graphics interchange layer.

The CPU reference fallback is deliberately simple and much slower than the
optimized Windows AVX provider. Native RTX source, kernel choices, benchmark
images and the current addon binary are not retuned by this portability work.
