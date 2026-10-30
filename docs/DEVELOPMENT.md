# Windows addon build and packaging recipes

[Project overview](../README.md) · [Documentation index](README.md)

Run every command below from the **repository root**, unless its example explicitly changes directory. See [BUILDING.md](../BUILDING.md) for broader toolchain/runtime setup and [the library guide](LIBRARY.md) for standalone integration.

## Build from source

The commands below build and validate the addon. For optional routes, use the
[dependency guide](../third_party/EXTERNAL_DEPENDENCIES.md) and [detailed build notes](../BUILDING.md).

### Requirements

Run the commands below in PowerShell from the repository root. Replace the
example SDK and model paths with your local paths.

| Requirement | Used for |
| --- | --- |
| Windows x64 | Building and running the Windows targets |
| [Python 3.11+](https://www.python.org/downloads/windows/) | Build scripts and automated tests |
| [Visual Studio C++ x64 Build Tools](https://visualstudio.microsoft.com/downloads/) and [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/) 10.0.26100 or newer | C++ compilation, linking, GPU upload declarations and Windows resource compilation |
| [NVIDIA NGX / DLSS SDK headers](https://github.com/NVIDIA/DLSS), supplied separately | Addon and native validation tools; `--ngx-sdk` must contain `nvsdk_ngx.h`, `nvsdk_ngx_defs.h` and `nvsdk_ngx_params.h` |
| [CUDA Toolkit 13.1](https://developer.nvidia.com/cuda-toolkit-archive), as tested locally | Native RTX kernels and the optional portable CUDA target |
| Python 3.12+ and [NumPy 2.5.2](https://pypi.org/project/numpy/2.5.2/) | Offline synthetic fixture generation for release validation |
| Pinned model DLL, supplied separately | Saved inference regression checks require the exact digest shown below |

Install the **Desktop development with C++** workload, including the x64 compiler
and Windows SDK. The build scripts locate Visual Studio through `vswhere` and
initialize its compiler environment automatically.

Published native kernels target **`sm_120`**. The current production builder also
supports experimental **`sm_75/86/89`** images for older RTX cards; see
[native architecture builds and validation](NATIVE_ARCHITECTURES.md).
Running the native validation steps requires compatible NVIDIA hardware; the
tested GPU is an RTX 5080. Compiling
the addon and its CPU/WARP tests does not require a model or physical NVIDIA GPU.

HIP, SYCL and Vulkan have additional optional dependencies. See
[external dependencies and tested versions](../third_party/EXTERNAL_DEPENDENCIES.md)
for installation links, SDK layouts and matching runtime libraries. Third-party
source notices are listed in [THIRD_PARTY.md](../THIRD_PARTY.md).

- Optional GPU tools: [ROCm / HIP](https://github.com/ROCm/TheRock/blob/main/RELEASES.md),
  [Intel LLVM / DPC++](https://github.com/intel/llvm/blob/sycl/sycl/doc/Releases.md),
  [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).
- Running in a game: [ReShade with addon support](https://reshade.me/) and the
  [Microsoft Visual C++ x64 runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

Python and NumPy are development tools; the compiled addon and runners do not
require Python, NumPy or Torch at runtime.

Measured memory ownership, process-memory definitions and the refinement/capture
regression scope are documented in [MEMORY_USAGE.md](../MEMORY_USAGE.md).

### Compile the addon

```powershell
python tools/build_addon.py addon_release --ngx-sdk C:\SDK\NGX\include --production-only
```

Output: **`build/addon_release/phoenix.addon64`**.

The builder runs 11 C++ test suites and two software D3D12 WARP suites before
compiling the addon. `--production-only` explicitly selects the release target.
The default also selects only that target when diagnostic addon source is absent.

This command builds the addon without embedding a runtime package. For a complete
embedded build, continue through the following sections.

### Build the kernels and runners

Build the selected native kernels with an explicit CUDA compiler path:

```powershell
python tools/production_build_native_kernels.py native_release --compiler "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" --jobs 4 --architectures sm_120
```

Output: **`build/native_release/`**, containing 41 kernel modules, their graph
and build manifest. PTX, cubins and other generated files stay under `build/`.
No pre-existing compiled kernels or extracted model files are required to compile.

Build the core runtime, CPU runner and validation helpers:

```powershell
python tools/build_portable_cpp.py runners_release --only cpu runtime broker runner device_query d3d12_transport runtime_contracts client_contracts
& ./build/runners_release/phoenix_cpp_runtime_contracts.exe
python tools/build_validation_tools.py validation_tools --ngx-sdk C:\SDK\NGX\include
```

Outputs are under **`build/runners_release/`** and **`build/validation_tools/`**.
This example includes the portable CPU route. Add other routes before packaging
if they are required in the final addon:

| Additional target in `--only` | Additional arguments / requirements |
| --- | --- |
| `cuda` | `--cuda-compiler PATH_TO_NVCC --cuda-arch sm_120` |
| `hip` | `--hip-sdk SDK_CORE_DIRECTORY --hip-arch TARGET_ARCHITECTURE` |
| `sycl` | `--sycl-compiler PATH_TO_CLANG_PLUS_PLUS` |
| `opencl_cpu_proxy` | Also needed when packaging the private Intel OpenCL CPU runtime |
| `vulkan_adapter vulkan_shaders` | `--vulkan-deps DIRECTORY_WITH_HEADERS_AND_GLSLANG` |

Keep the core targets in `--only` when adding optional targets. The final package
contains only the targets you built and the dependencies you explicitly supply.

### Validate and package

Compilation and inference validation are separate steps. Embedding requires a
validated portable runtime package and a validated native kernel bundle.

Install the offline fixture dependency using Python 3.12 or newer:

```powershell
python -m pip install numpy==2.5.2
```

Create **`build/weight_source.json`** with an absolute path to your separately
obtained model DLL. The checked-in regression data uses this exact DLL digest:

```json
{
  "kind": "nvidia_dll",
  "path": "C:/Models/nvngx_dlssnr.dll",
  "sha256": "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e"
}
```

The saved regression checks require the matching DLL; changing the digest alone
does not make a different model compatible with those reference outputs.
Check your DLL before creating the configuration; a newer game's DLL may differ:

```powershell
$modelPath = 'C:\Models\nvngx_dlssnr.dll'
$expectedDigest = 'e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e'
if ((Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedDigest) {
    throw 'The DLL does not match the saved regression model.'
}
```

For the CPU runtime package built above:

```powershell
python native_runtime/package_build.py runners_release runners_package
& ./build/runners_release/phoenix_cpp_client_contracts.exe ./build/runners_package ./build/weight_source.json ./build/client_contracts.json
python tools/prepare_validation_fixtures.py ./build/runner_inputs --runtime ./build/runners_package --weight-source ./build/weight_source.json --routes ./tests/fixtures/cpu_routes.example.json
python tools/validate_runners.py --runtime ./build/runners_package --runner ./build/runners_release/phoenix_cpp_runner.exe --jobs ./build/runner_inputs/jobs --output ./build/runner_validation
python tools/validate_native.py --bundle ./build/native_release --probe ./build/validation_tools/native_inference_probe.exe --weight-source ./build/weight_source.json --output ./build/native_validation
```

Each command must succeed before proceeding. The tools generate synthetic inputs
locally and compare outputs with saved Phoenix DLSS5 regression hashes. Native validation
covers eight saved extents, including 1920×1080, 6K, 8K and odd sizes on both
sides of the automatic kernel-selection threshold. Two modules contain default
and large-input entries; validation must exercise both choices before packaging.

For optional backends, provide their vendor libraries using the package builder's
repeatable `--dependency PATH_TO_DLL` option. For SYCL CPU, also supply
`--opencl-cpu PRIVATE_ICD_DIRECTORY` and build `opencl_cpu_proxy`. Discover IDs with:

```powershell
& ./build/runners_release/phoenix_cpp_devices.exe ./build/runners_package
```

Add `private-cpu` only when that package contains the private Intel CPU ICD and
OpenCL proxy. Missing optional providers are expected when their targets were
not built. Supply the returned IDs in a local route mapping to `--routes`.
See [BUILDING.md](../BUILDING.md) for route names and validation details.
Compiling an optional provider alone does not enable it in the validated package.

### Build the embedded addon

After both validations pass:

```powershell
python tools/build_addon.py addon_embedded --ngx-sdk C:\SDK\NGX\include --production-only --embed-runtime ./build/runners_package --embed-native ./build/native_release
& ./build/validation_tools/embedded_runtime_tests.exe ./build/addon_embedded/phoenix.addon64 ./build/resource_cache ./build/resource_test.json
```

Final output: **`build/addon_embedded/phoenix.addon64`**.

The runner libraries and GPU kernels are embedded in this file. The NVIDIA DLSS5 model
DLL remains separate. A game installation also needs compatible ReShade with
addon support and a local `Phoenix.json` configuration; see [BUILDING.md](../BUILDING.md).

### Other graphics APIs and 32-bit games

The experimental graphics bridge builds `Phoenix.addon32` and `Phoenix.addon64`,
with a private x64 helper embedded. Both cover D3D9, D3D10/10.1, D3D11,
D3D12, OpenGL and Vulkan; x64 D3D12 retains its direct route. All APIs return
matching pixels in the x64 matrix. x86 transport/sharing and a real D3D9 game
are checked; the remaining x86 game/API checks are pending. These final-color
bridges lack engine motion/depth inputs, and readback adds latency.
Matched-frame display is the default: hold the last completed input/output pair
until the next result is ready, so newer movement cannot show under older masks.
The panel offers experimental live-frame editing for D3D9/D3D11.
Both bitnesses reuse the main controls and timing UI. The thin addon obtains
devices, validation and timing data from its x64 helper; controls requiring
missing engine inputs remain visible with a capability explanation.
See [build, installation, latency and validation details](GRAPHICS_COMPATIBILITY.md).

An opt-in [execution mechanism selector](EXECUTION_MECHANISMS.md) also wires
the matching-bitness portable runtime into the existing fenced pixel transport.
It keeps native D3D12, the x64 helper, shared textures and staging routes
configurable. Automatic routing is unchanged; the alternate local x86 Vulkan
path remains experimental pending game-presentation validation.

### Shared processing pipeline

The addon, Phoenix Video Player and DLSS Everything share ordered
processing recipes, motion estimation, scene resets, monitoring/recovery and
optional native NIS, tiny 2x scaling and multiscale fusion. Models remain external.
The shared x64 portable frame worker also exposes intensity/style controls.
Both models have a real capture-worker neural/appearance pixel oracle against
the public runtime; zero intensity restores the source. Diagnostics distinguish
the capture host from the native neural workers and verify worker retirement.
The player package includes its native SDK, kernels and pinned media helpers;
Everything supplies the centrally installed model path.

[1080p pipeline costs, memory and native-pixel face comparisons](benchmarks/current/pipeline/README.md)
measure both Phoenix models and the optional stages. These synchronous host-API
measurements include transfers and are separate from the existing GPU-resident
[Phoenix DLSS5 / NVIDIA DLSS5 benchmarks](PERFORMANCE.md). Fusion can soften detail, and the current tiny
scaler has visible artifacts and a high cost; both remain experimental.

Native chain, fusion, cache/live segment and source-to-Present API checks pass
on the RTX 5080. Portable color/fusion stages also have checks on CPU, CUDA and
Vulkan on NVIDIA/AMD, with independent histories and unchanged ordinary output.
The x64 final-frame portable worker and player can also use the shared D3D12
stage graph: independent processing/output extents, NIS/tiny scaling, motion,
scene resets and coordinated latency control, with native portable neural
workers per pass/branch. Everything and the player offer installed processing
devices. These adapters retain host transfers and are separate from the
equal-extent C API contract. VSR and OptiScaler are deferred. The
[implementation ledger](PIPELINE_IMPLEMENTATION.md) records exact evidence
and remaining work; this checkpoint does not establish every new stage on every
existing backend.

[The pipeline guide](PROCESSING_CHAIN.md) separates the add-on features
from the companion applications and the retained development measurements.
The release embeds updated portable runtime and native SDK companions, with
fresh CPU/software checks and no new GPU qualification. Flow/scene/pipeline-age controls require the
local final-frame D3D12 bridge or the x64 shared portable frame worker; other insertion routes retain their supplied
guides/reset flags and neural-budget controls. Experimental x86 native bridges
reject custom chains and frame services they cannot execute. The existing portable `phoenix_frame` CLI also accepts
the [color/fusion metadata contract](LIBRARY.md#portable-color-and-fusion-stages).

### Tests and source checks

```powershell
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s tests -p '*tests.py'
python tools/audit_release.py --git-visible
```

For a source archive without a `.git` directory, omit `--git-visible`.

Use a **fresh output name** for each build or validation run. Builders deliberately
refuse to overwrite existing output directories. Compiler logs, test reports and
validation manifests are written alongside their generated outputs under `build/`.
The addon build report records compilation, contract tests and final artifact
verification separately. A successful linker exit alone does not complete a build.
If the output is missing or locked, inspect `report.json`, the build logs and
your security product's history. Packaging is complete only after the builder
and embedded-resource check both succeed.
