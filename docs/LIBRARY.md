# Standalone Phoenix library (experimental)

Phoenix can be embedded without ReShade through the C interface in
[`native_runtime/phoenix.h`](../native_runtime/phoenix.h). The addon remains the
main, optimized Windows integration. This standalone interface currently uses
the portable graph and host-plane frame boundary; **it does not expose the fast
D3D12/NVAPI native RTX path**. Its performance must not be inferred from the
native RTX charts.

## Platform status

| Platform | Implementation | Validation available here |
| --- | --- | --- |
| Windows x64 | C DLL; portable CPU and Vulkan; existing optional CUDA/HIP/SYCL provider ABI retained | CPU/operator/C-interface tests and full-model frame |
| Linux x64 | C shared library; portable CPU and Vulkan build paths | Built and CPU tests/full-model frame run under Ubuntu 22.04 WSL |
| macOS, Apple silicon | C dynamic library; Metal compute backend and portable CPU | Apple compiler/MSL and CPU suites pass in macOS 14 ARM64 CI; full-model GPU execution unvalidated |
| Android ARM64 | NDK C shared library; portable CPU and Vulkan | Cross-compilation; no phone GPU or full-model execution certified |
| Android x86_64 emulator | NDK library with a JNI/OpenGL game-frame harness | Full-model CPU/Vulkan execution; three Vulkan image sizes match Windows portable Vulkan byte for byte; physical phones remain untested |

Metal is an initial correctness implementation, with resident buffers, ordered
GPU dispatches, packed FP8 weight decoding and explicit half/FP8 publication.
Its shader is generated from the shared operator reference, rather than a
separate arithmetic implementation. Matrix operations currently use scalar
GPU work items; optimized Metal matrix tiles/MPS are future work. It does not
use the Apple Neural Engine. GPU time covers the complete submission; there is
no fabricated per-layer breakdown.

Vulkan reuses the existing desktop provider and shaders. It needs Vulkan 1.3,
shader Int64/Float16, 16-bit storage, buffer device addresses, scalar block
layout, the Vulkan memory model and support for a 32-lane subgroup. Cooperative
matrices are optional and used only when their shapes are advertised. Basic
Vulkan availability on a phone does not guarantee these requirements. Unsupported
devices are rejected; no hidden CPU fallback is reported as GPU inference.

Devices without cooperative matrices now use an ordered 32×32 shared-memory
matrix tile. Packed expanding projections and large 128-to-32 contractions
use coalesced weight loads; other shapes retain the simpler tile. This changes
data reuse, not precision or model weights. Matrix-capable devices retain their
existing accelerated route. Status reports `vulkan_matrix_acceleration` and
`vulkan_fallback_tile` so these paths can be distinguished.

Advanced comparison overrides are `vulkan_tiled_fallback=false` (the previous
8×16 fallback) and `vulkan_coalesced_fallback=false` (the simple 32×32 tile).
The tested `vulkan_scalar_attention=true` alternative fuses local attention
without matrix extensions, using 6 KiB of workgroup storage. It remains opt-in:
operator/model output checks pass, but full-call speed gains were inconclusive.
These options do not disable an available matrix-accelerated route.

Linux CUDA and a Mac/Android game overlay are not implemented by this port.
Applications must supply their own image/graphics integration. Host-plane
copies and C++ preparation/resolve still exist; no zero-copy texture API is
promised. The CPU reference is a portability/correctness fallback, not the
optimized AVX Windows CPU runner and not a real-time performance claim.

## Build a library

Requirements: CMake 3.22+, a C++20 compiler, Python 3.10+ for shader generation
only, and the platform SDK/toolchain. Generated sources/binaries stay under
`build/`. No Python runtime is required by an installed library.

- [CMake](https://cmake.org/download/), [Python](https://www.python.org/downloads/).
- Windows: [Visual Studio C++ build tools](https://visualstudio.microsoft.com/downloads/).
- Linux: GCC 11+ or a current Clang, development headers and pthread/dl libraries.
- Mac: [Xcode](https://developer.apple.com/xcode/) with the Metal compiler, macOS
  13+ recommended; Apple silicon required by the current Metal backend.
- Android: [NDK](https://developer.android.com/ndk/downloads), tested cross-compiler
  NDK **27.3.13750724**, ARM64, API **28** minimum for the library. Vulkan 1.3
  support is a separate device/driver requirement.

Windows/Linux CPU build:

```sh
cmake -S native_runtime -B build/standalone -DCMAKE_BUILD_TYPE=Release -DPHOENIX_ENABLE_METAL=OFF
cmake --build build/standalone --config Release --parallel 4
ctest --test-dir build/standalone -C Release --output-on-failure
cmake --install build/standalone --config Release --prefix build/sdk
```

Mac build (Metal enabled by default on macOS):

```sh
cmake -S native_runtime -B build/metal -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0
cmake --build build/metal --parallel 4
ctest --test-dir build/metal --output-on-failure
cmake --install build/metal --prefix build/sdk
```

The Mac build compiles the generated MSL with `xcrun metal` as a build check.
`metal_operators` also executes it on an Apple silicon GPU and compares with
the shared arithmetic reference. Run it before treating a Mac build as usable.
CI compiles Metal and enumerates devices. It runs GPU operator tests only if the
host exposes a supported Metal device; otherwise it reports them as not executed.

Android cross-build (substitute your NDK path):

```sh
cmake -S native_runtime -B build/android -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/ndk/27.3.13750724/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_shared -DPHOENIX_ENABLE_METAL=OFF
cmake --build build/android --parallel 4
cmake --install build/android --prefix build/android-sdk
```

Package both Phoenix `.so` files and the NDK's `libc++_shared.so` in the app's
ARM64 native library directory. This is a C/NDK interface; a Kotlin/Java app
needs a JNI wrapper. No APK, UI, camera integration or game injection is supplied.
Model files must be readable filesystem paths; Android content URIs need to be
copied/mapped by the host app first. An app's loader may require its native-library
directory as `runtime_root` and APK settings that extract native libraries.

### Enable Vulkan on Linux/Android

Install [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) version
**1.4.365** or newer, a system/NDK Vulkan loader, and host
[glslang](https://github.com/KhronosGroup/glslang) **16.6.0** or newer. The NDK's
bundled headers alone are older than declarations used by the existing provider;
pass the separately installed headers. `PHOENIX_GLSLANG` must run on the build
computer, including during Android cross-compilation.

Add these options to the relevant configure command:

```sh
-DPHOENIX_ENABLE_VULKAN=ON \
-DPHOENIX_VULKAN_HEADERS=/path/to/Vulkan-Headers/include \
-DPHOENIX_GLSLANG=/path/to/glslang
```

Windows CMake builds additionally need the Vulkan loader import library from an
installed Vulkan SDK (`-DPHOENIX_VULKAN_LOADER=/path/to/vulkan-1.lib`). The established
Windows build in [BUILDING.md](../BUILDING.md) remains available.

The generated SPIR-V and provider go in `lib/vulkan/` after installation
(`bin/vulkan/` on Windows). Keep them beside the backend's runtime root. Enumerate
devices with `phoenix_frame --vulkan-devices /absolute/runtime/root`. Configure
`PHOENIX_VULKAN_TEST_DEVICE=vulkan:<32-character-UUID>` to add the device operator
suite to CTest. Shader and provider generation is shared with the Windows build.

## Embed it

Installed CMake packages provide `Phoenix::Runtime`:

```cmake
find_package(Phoenix CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE Phoenix::Runtime)
```

Use `-DCMAKE_PREFIX_PATH=/path/to/sdk`. Plain C callers can include `phoenix.h`
and link `phoenix_runtime_cpp.lib` on Windows or `libphoenix_runtime_cpp` on Unix.
Deploy its selected backend library too. Build backend plugins with the same C++
toolchain as the runtime; the public application boundary is C, the internal
backend plugin boundary is C++.

```c
#include <phoenix.h>
/* Allocate RGB float32 buffers for width*height pixels first. */
void *session = phoenix_runtime_create(config_json, width, height);
if (!session) { /* copy phoenix_runtime_error() */ }
else {
    int result = phoenix_runtime_frame(session, color, NULL, NULL, NULL,
                                       "{\"reset\":true}", output);
    if (result) { /* copy phoenix_runtime_error() */ }
    phoenix_runtime_destroy(session);
}
```

Input/output are tightly packed RGB32F, **model proxy values in 0..1**, same
dimensions, not arbitrary HDR game buffers. Motion is RG32F in pixel units;
depth and validity are R32F. Optional planes require matching `has_motion`,
`has_depth`, `has_history_validity` metadata; validity additionally requires motion.
The application owns these arrays and must provide their documented capacities.
Calls are synchronous. Reuse a session for temporal history; reset when cutting
scenes. Serialize its calls and never destroy it during a frame call.
The current Vulkan provider permits one active Vulkan session per process;
creating a second fails explicitly. CPU and Metal sessions have independent
backend state. Separate processes are an option for simultaneous Vulkan streams.

Configuration example (absolute paths, local model supplied separately):

```json
{
  "backend": "metal",
  "device": "metal:default",
  "runtime_root": "/absolute/path/to/sdk/lib",
  "allow_unvalidated": true,
  "cpu_threads": 0,
  "weight_source": {
    "kind": "nvidia_dll",
    "path": "/absolute/path/to/nvngx_dlssnr.dll",
    "sha256": "REPLACE_WITH_THE_ACTUAL_LOWERCASE_SHA256"
  }
}
```

For CPU use `backend: cpu`, `device: cpu`. For Vulkan use `backend: vulkan` and
the enumerated `vulkan:<UUID>`. Mac devices can be listed with
`phoenix_frame --devices /absolute/path/to/sdk/lib`; prefer their registry IDs
over `metal:default` when multiple GPUs are present. Unvalidated devices require
explicit opt-in; a device's presence is not evidence of model parity.
On Windows use absolute JSON paths such as `C:/Phoenix/sdk/bin`.

The NVIDIA Vulkan route automatically enables the workgroup FP8 engine only
when the selected device exposes the required matrix shapes, tensor addressing
and conversion features. The configuration switches `vulkan_workgroup_fp8`,
`vulkan_workgroup_shape_tiles`, `vulkan_fused_feed_forward` and
`vulkan_projected_qkv` default to `true`; setting any to `false` disables that
optimization. Other devices keep their existing engines. Legacy packed weight
codes incompatible with hardware FP8 automatically use the established decoder.
These switches select execution paths for the same model, without adding a
second weight copy or changing its precision/publication rules.

The model DLL is read as **binary data**, never executed by this runner. Unix
hosts parse its PE resource tree to find `RCDATA/WEIGHTS_HT`, authenticate the
file and decode the tensor records. Disk models use the existing
`dlss5-phoenix-model/1`, `phoenix-network37/1` manifest/layout. This supports
compatible replacement tensors; it does not import arbitrary ONNX/PyTorch
architectures. No NVIDIA DLL or extracted model is bundled in the source/SDK.

Per-frame controls:

```json
{
  "reset": false,
  "passes": 1,
  "repeats": [1,1,1,1,1,1,1,1],
  "controls": {
    "intensity": 1.0,
    "local_tone": 1.0,
    "local_structure": 1.0,
    "skin_structure": -1.0,
    "auto_mask": true,
    "style": 0,
    "seed": 0
  }
}
```

Passes are 1..4; each of the eight global block repetition counts is 0..4.
Changing repeats rebuilds the graph and clears history; it is not a free
per-frame switch. Intensity/tone/structure are 0..2, skin is -1..2, style 0..2.
The host can adapt repeats/passes or processing resolution to its own FPS/time
budget using returned timings. This ABI does not include the addon's automatic
budget controller or upscaler integration. Resize by creating a new session.

`phoenix_runtime_status` reports GPU/CPU timing scope, resident weight/scratch
bytes, sampled host capacities and history usage. These allocations are not
total process RAM/VRAM. `phoenix_runtime_error` is thread-local; copy it before
another call. Query status size, allocate that many bytes, then retrieve it.

## Standalone frame tool

`phoenix_frame JOB.json NEW_OUTPUT.rgb32f` loads a configuration, processes one
reset frame and prints JSON status. Its input file contains little-endian RGB
float32 values (three floats per pixel); optional motion/depth/validity planes
are supplied the same way. Output must be a fresh path.

```json
{
  "width": 640,
  "height": 360,
  "color": "/absolute/path/input.rgb32f",
  "config": { "backend": "cpu", "device": "cpu",
    "runtime_root": "/absolute/path/to/sdk/lib",
    "allow_unvalidated": true,
    "weight_source": { "kind": "model_folder", "path": "/absolute/path/model" }
  },
  "metadata": { "passes": 1 }
}
```

See [port validation](PORT_VALIDATION_2026-10-14.md) for the exact local checks
and remaining hardware requirements. Source and standalone packages retain the
project/third-party licenses; vendor models and SDK binaries remain external.
