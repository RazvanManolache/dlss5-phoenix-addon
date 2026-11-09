# Native CUDA executor for Linux (SM120 preview)

Linux can execute Phoenix's production native model schedule without D3D12 or
NVAPI. This executor shares the authenticated model loader, packed weights,
resolution planner, launch argument relocation and 41 production kernel sources
with Windows. Original runs 154 neural launches; Lightweight FP8 runs 78.
CUDA streams/graphs provide ordered dispatch and CUDA texture/surface objects
replace the Windows descriptor/sampler ABI. No Windows DLL code is executed.

The current scope is **SM120, SDR final-color input, reset on every frame**.
Both models are selectable. Motion, temporal reuse, advanced V3 controls,
adaptive depth, multistage pipelines, model hot-swapping and other GPU
architectures are rejected or unavailable in this frontend. The portable CUDA
backend remains separately selectable when its broader host-frame API is needed.
The launcher never silently substitutes a backend or GPU.

## Select the executor

Use a complete Linux package containing `libphoenix_native_cuda.so` and the
model-free `lib/native-cuda` bundle. Configure an external model as in
[the Linux guide](LINUX.md), with these changes:

```json
{
  "backend": "native-cuda",
  "device": "cuda:REPLACE_WITH_32_HEXADECIMAL_UUID_DIGITS",
  "allow_unvalidated": true,
  "weight_source": {
    "kind": "nvidia_dll",
    "path": "/absolute/path/to/compatible/nvngx_dlssnr.dll",
    "sha256": "REPLACE_WITH_AUTHENTICATED_DLL_SHA256",
    "model": "small-fp8"
  }
}
```

Select `original` for the full model. The CUDA device UUID uses the same 32
hexadecimal device UUID digits, with the `cuda:` prefix. The optional
`--game-device vulkan:<UUID>` still selects the rendering/presentation GPU
independently. The launcher selects the native runtime library and supplies its
relocatable runtime root. The default kernel bundle is adjacent to that library;
`native_bundle` can explicitly select an absolute external bundle path.

The native runtime requires the NVIDIA driver (`libcuda.so.1`). The CUDA runtime
is linked statically; users do not need the toolkit. GPU modules and graph hashes
are verified before execution. CUDA and D3D12 bundles have different boundary
ABIs: the loader checks this explicitly. Never replace a Windows kernel bundle
with a Linux CUDA bundle.

Automatic masking defaults on, inverted depth is disabled, and tone/structure
default to 1, matching the corrected native comparison settings. Optional
`local_tone`, `local_structure`, `skin_structure` and `auto_mask` are accepted.
Unknown configuration controls are rejected. Without engine motion, every game
frame resets sampling/history; this is not a claim of temporal reconstruction.

## Transport and timing

The Vulkan frontend uses an optional bounded RGBA8/BGRA8 runtime entry point
from [phoenix_linux.h](../native_runtime/phoenix_linux.h). Its byte capacity,
stride, extent, format and reset flag are checked before GPU work. Pixels are
snapshotted through pinned host staging; conversion to/from the native float
textures runs on CUDA. Alpha and encoded SDR values are preserved. Caller memory
is updated only after successful execution and finite-output checking.

This removes full-frame RGB32F transport and CPU conversion from the game hook.
It **still uses host transport**, including the Vulkan readback/upload. It does
not provide GPU-to-GPU zero-copy or repair the NVIDIA presentation route on the
tested desktop. The bounded float32 V2 API remains available, including strided
and aliased input/output. It requires exactly `{"reset":true}` metadata; unsupported
optional planes/controls fail before writing caller output.

GPU event timing covers counter initialization and the complete native
stem/model/head. Synchronous API timings also include staging, transfers and
conversion. Game hook time additionally includes Vulkan transport. Presentation
intervals include game work and presentation pacing. These measurements must
not be presented as the same metric. Current measured results and their limits
are recorded in [the 0 A.D. report](LINUX_0AD_VALIDATION.md).

## Build and validate

Use an external CUDA Toolkit 13.1 or newer; the development run used 13.1.80.
No driver install, clock/power changes or global layer registration is required.
Enable both optional CUDA paths in an otherwise complete Vulkan-layer build:

```sh
cmake -S native_runtime -B build/linux-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPHOENIX_ENABLE_METAL=OFF \
  -DPHOENIX_ENABLE_VULKAN=ON -DPHOENIX_BUILD_LINUX_LAYER=ON \
  -DPHOENIX_VULKAN_HEADERS=/path/to/Vulkan-Headers/include \
  -DPHOENIX_GLSLANG=/path/to/glslang \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DPHOENIX_ENABLE_CUDA=ON -DPHOENIX_ENABLE_NATIVE_CUDA=ON
cmake --build build/linux-native --parallel 4
cmake --install build/linux-native --prefix build/linux-prefix
python3 tools/package_linux.py --prefix build/linux-prefix --output artifacts/current/linux --version 1.0.1-linux-preview
```

For the same pinned private Linux x86_64 SDK used by CI, run
`python3 tools/fetch_cuda_toolchain.py --output /path/to/fresh/cuda-13.1`.
It verifies fixed archive sizes/SHA-256 pins and writes only that new directory;
it does not install a driver or alter the system PATH. An existing toolkit may
be used instead. CI compiles the CUDA executors and runs software contracts;
it does not claim GPU/game validation from a CPU runner.

Set `PHOENIX_CUDA_LICENSE` if the toolkit license is stored outside its root.
The build includes that notice with statically linked CUDA runtime payloads.
The native kernel builder uses the production register/assembler recipes and
CUDA texture objects. Its manifest identifies source hashes and compiler version;
`passed` means compilation/integrity checks passed, not GPU certification.

Software-only tests are selected with `ctest -R '^linux_'`. GPU tests are explicit:

```sh
python3 tests/linux/native_cuda_contracts.py --package /path/to/package \
  --config /path/to/native.json --image /path/to/input.png --report /path/to/contracts.json
python3 tools/profile_linux.py --package /path/to/package \
  --config /path/to/native.json --image /path/to/input.png --rgba8 --output /path/to/profile.json
```

These development scripts require Pillow. Profiling excludes ten initial calls
by default, restores the input before every packed call, hashes outputs outside
the timer, and reports model GPU events separately from synchronous API time.
For CUDA memory validation, run the explicit contract script under
`compute-sanitizer --tool memcheck --error-exitcode 99`. Vulkan validation checks
the presentation frontend; it does not validate CUDA memory accesses.

## Low-level references investigated

Separate read-only clones were inspected; no code or instruction tables from
these projects are included in Phoenix or its package:

- [blackwell-isa](https://github.com/kacper-daftcode/blackwell-isa),
  `c374bad146afd10eb41aa1ccd793c09422e2ddfb`, MIT: SM120 instruction encodings,
  scheduling metadata and provenance. Useful for inspecting emitted tensor-core
  code and register/stall choices. Encoding round trips alone do not establish
  instruction semantics or speed on the RTX 5080.
- [cubit](https://github.com/kacper-daftcode/cubit),
  `9d7fad29cf23ee6e0a12811194d8460ae029d4f3`, MIT: SASS assembler/disassembler,
  frozen cubin reconstruction, scheduling and stall-floor experiments. Candidate
  instruction changes need a byte-exact unchanged round trip, numerical tests
  and per-kernel timing before adoption. SM100 datacenter tensor-memory recipes
  are not automatically SM120 recipes.
- The local `sovereign-intelligence` checkout,
  `486e91806e7d2d1f56b8489e917eafe6d2791e52`, and its
  [CUDApp source](https://www.aienos.com/rad/z4RxD8vXigmWg61nM8miQ8DVPAXEP.git),
  `291c7a58bd53eff4cb544ec0a91eecf63b95cdaa`, Apache-2.0: persistent packed
  scratch, cache policy, banded GEMM tile traversal and launch-cost measurements
  are useful ideas. The cited launcher evidence is scoped to GB10/SM121 and
  driver 580.173.02; it is not an RTX 5080/595.99.02 executor. This work reuses
  Phoenix's native kernels through CUDA rather than depending on that driver ABI.

The direct Vulkan/CUDA marker probe is retained at
[tests/linux/cuda_launch_probe.cpp](../tests/linux/cuda_launch_probe.cpp).
It demonstrated `VK_NV_cuda_kernel_launch` on the RTX 5080 without NVAPI or a
CUDA toolkit, but it is a capability probe, not the production model executor.
The current executor uses CUDA's model/module APIs and reusable graph dispatch.
