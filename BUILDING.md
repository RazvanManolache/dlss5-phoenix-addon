# Build Phoenix DLSS5

<!-- current-two-model-evidence -->
Current two-model support and platform requirements are listed in [the support matrix](docs/SUPPORT.md). The [current benchmark report](docs/benchmarks/current/README.md) includes reproducible preparation/run commands and identifies the measured development binaries; it is separate from published packages.
<!-- /current-two-model-evidence -->

For the standalone C library, Linux/Android builds and experimental Apple silicon
Metal backend, see [the library build and integration guide](docs/LIBRARY.md).
The instructions below build the Windows ReShade addon and its existing runners.

Build from this repository with Python 3.11 or newer, Visual Studio C++ x64 build tools and Windows SDK 10.0.26100 or newer. Python is used for building; the compiled runners do not require it at runtime.

Public Windows SDK builds use MSVC deterministic path mapping so diagnostic
source locations do not disclose the builder's workspace. When building the
portable runners with `tools/build_portable_cpp.py` for distribution, also set
`CL` to `/experimental:deterministic /pathmap:C:\SourceRoot=PROJECT_SOURCE`,
replacing the example root with the parent of your checkout. Run the package
privacy checks on the final binaries, not only on their source archive.

Provide NVIDIA NGX SDK headers separately. They are not included here. Native RTX kernels also require a CUDA toolkit supporting the architecture in `src/native/model/production_kernel_contracts.json` (currently `sm_120`).

See [external dependencies and tested versions](third_party/EXTERNAL_DEPENDENCIES.md)
for SDK installation sources, required runtime libraries and optional backend components.

Run `python tools/audit_release.py --git-visible` in a Git checkout to check that
every source-package file is in Git's index and that model/binary payloads are
excluded. The auditor requires a Git checkout. Local editor configuration is
excluded even if Git lists it. C++ architecture/layout sources under
`native_runtime/models/` and `src/native/model/models/`, plus `models/README.md`,
are included; downloaded/extracted model data remains forbidden. Commit the
audited source changes before preparing a release archive.

## Addon and tests

From the repository root in PowerShell:

```powershell
python tools/build_addon.py addon_release --ngx-sdk C:\SDK\NGX\include --production-only
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s tests -p '*tests.py'
```

Use a fresh temporary output directory for a requested build. Keep only the requested current deliverables in `artifacts/current` and remove the temporary build afterwards; do not accumulate dated builds or duplicate packages. The addon builder runs 11 C++ contract suites and a D3D12 WARP capture test before compiling `build/addon_release/phoenix.addon64`. WARP tests capture allocation and fence retirement without loading a model or running neural inference. `--production-only` explicitly omits the separate diagnostic addon. When its source is absent, the default also builds only the production addon.

## Native RTX kernels

```powershell
if (-not $env:CUDA_PATH) { throw 'Set CUDA_PATH to your installed CUDA Toolkit directory.' }
python tools/production_build_native_kernels.py native_release --compiler "$env:CUDA_PATH/bin/nvcc.exe" --jobs 4
```

This builds 41 production kernel modules and their runtime graph/manifest under `build/native_release`. Two modules contain both default and large-input entry points; each network instance creates only the selected 41 function handles. The repository contains their complete source dependency set; historical tuning profiles are not required.

All 41 kernels build from checked-in CUDA source and local helper headers. The input preparation and output reconstruction stages compile their `.cu` files to `kernel.ptx` inside the build directory, then assemble that generated PTX to `kernel.cubin`. The builder preserves independent D3D12 texture/sampler addressing and the selected 128-register budget for these two stages. These settings are recorded in `production_kernel_contracts.json`; they are required to reproduce the selected production code.

PTX and cubins are generated artifacts and are excluded from Git and source packages. Rebuilding requires no historical `compact_neural_render` folders, private source includes, extracted model, or pre-existing PTX. Model loading remains a separate runtime step.

The catalogue also records the selected assembler optimization level where it
differs from the default. These are production build requirements, not a tuning
menu. Device debug tables remain in generated build files and are omitted from
shipping cubins.

The catalogue includes the two large-input entry names and their selection
threshold. Global attention and the 512-channel attention projection switch to
separately compiled 128-register entries at 18,662,400 neural input pixels
(5760×3240 area). Smaller inputs retain their original entries. Selection uses
the actual model input, not the output/display resolution; arbitrary aspect
ratios use the same area rule. The three vectorized 64-channel stages retain
their selected spill-free register budgets at every size. Both entries are
built from the same checked-in operation source and stay in the same addon.

The native runner queries GPU upload-heap support on the selected device. When
available, immutable weights are written directly into GPU memory before the
first submission. Otherwise it uses a staging upload and releases that resource
only after the submission fence completes. Older devices and Windows runtimes
can use this fallback; the newer SDK is needed for the build declarations.

## Portable C++ runners

The CPU, runtime and broker need only MSVC and the Windows SDK:

```powershell
python tools/build_portable_cpp.py runners_release --only cpu runtime broker runner device_query d3d12_transport runtime_contracts client_contracts
& ./build/runners_release/phoenix_cpp_runtime_contracts.exe
```

Additional targets are `cuda`, `hip`, `sycl`, `vulkan_adapter`, `d3d12_transport` and `vulkan_shaders`. Select them with `--only` alongside the core targets. CUDA uses `--cuda-compiler` and `--cuda-arch`; HIP requires `--hip-sdk` and the intended `--hip-arch`; SYCL requires `--sycl-compiler`. Vulkan shaders require `--vulkan-deps` pointing to a directory containing `headers/include/vulkan/vulkan.h` and `glslang/bin/glslang.exe`. Vendor toolchains and runtime dependencies are supplied separately.

## Runtime packaging

For the optional NVIDIA DLSS5 NR comparison, install the x64 addon as
`Phoenix_nvngx.dll.addon64`. This NVIDIA DLSS5 runtime checks its calling module's
path for `nvngx.dll`; installing the same bytes as `Phoenix.addon64` allows
Phoenix DLSS5 inference but makes NVIDIA DLSS5 NR initialization return `0xBAD00002`.
Keep only one active copy of the addon. Renaming NVIDIA's model DLL does not
satisfy this calling-module requirement.

A successful compilation does not establish inference correctness or make a
self-contained game install. The validation tools and synthetic input recipes
are checked in; they do not need the former development folder or saved captures.
The model DLL remains an explicit external input. The supplied regression hashes
cover the pinned model listed in `third_party/EXTERNAL_DEPENDENCIES.md`.

Build the offline native probe and embedded-resource test:

```powershell
python tools/build_validation_tools.py validation_tools --ngx-sdk C:\SDK\NGX\include
```

Create `build/weight_source.json` locally, for example:

```json
{"kind":"nvidia_dll","path":"C:/Models/nvngx_dlssnr.dll","sha256":"e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e"}
```

Assemble and validate a CPU runtime package:

```powershell
python native_runtime/package_build.py runners_release runners_package
& ./build/runners_release/phoenix_cpp_client_contracts.exe ./build/runners_package ./build/weight_source.json ./build/client_contracts.json
python tools/prepare_validation_fixtures.py ./build/runner_inputs --runtime ./build/runners_package --weight-source ./build/weight_source.json --routes ./tests/fixtures/cpu_routes.example.json
python tools/validate_runners.py --runtime ./build/runners_package --runner ./build/runners_release/phoenix_cpp_runner.exe --jobs ./build/runner_inputs/jobs --output ./build/runner_validation
python tools/validate_native.py --bundle ./build/native_release --probe ./build/validation_tools/native_inference_probe.exe --weight-source ./build/weight_source.json --output ./build/native_validation
```

`prepare_validation_fixtures.py` requires NumPy only for offline generation.
It reconstructs the random inputs, motion, depth and validity masks and verifies
every input checksum before creating jobs. `validate_runners.py` requires a
saved output hash for every case, checks the full case count and module audit,
and refuses an empty job set. A success receipt enables only routes actually
tested. Failure does not produce a successful validation manifest.

Before running these checks, verify the DLL with `Get-FileHash -Algorithm SHA256`
as shown in [README.md](README.md). A different DLL, including a newer release,
does not match the saved regression outputs merely by changing the configured digest.

For more routes, build their targets and add matching vendor runtimes when packaging.
Discover device IDs using:

```powershell
& ./build/runners_release/phoenix_cpp_devices.exe ./build/runners_package
```

Append `private-cpu` only for a package built with `opencl_cpu_proxy` and
`--opencl-cpu PRIVATE_ICD_DIRECTORY`. That option requires both the Intel CPU ICD
and OpenCL proxy under the package's `opencl_cpu` directory. Unbuilt optional
providers may appear under `errors` without invalidating available device IDs.
Pass a local JSON mapping names (`cpu`, `cuda`, `hip_amd`, `sycl_cpu`, `vulkan_amd`,
`vulkan_nvidia`) to those IDs as `--routes`. The reference hardware is documented
in `tests/fixtures/runner_regressions.json`; another GPU/toolchain may need its own
reviewed numerical reference. A mismatch must be investigated, not bypassed.

`validate_native.py` generates inputs for eight saved regression extents, runs 128
frames per extent and compares reset and history readbacks with saved Phoenix DLSS5
native output hashes. It checks debug errors, GPU execution and kernel retirement
before marking the bundle validated. Cases include 6K, 8K, and 5760×3239/5761×3241
on opposite sides of the selection threshold. Both compiled branches must be
exercised; packaging also verifies that the validated entry/threshold metadata
has not changed. This is Phoenix DLSS5 regression equivalence,
not equivalence to NVIDIA's renderer. `--cpu-only` on the standalone native probe
checks geometry only and cannot satisfy inference validation.

Source-built native bundles generate launch plans on demand for the requested
resolution, including odd dimensions, portrait and ultrawide images. Feature
tensors are padded internally; the displayed image keeps its active dimensions.
The runner caches the four most recently used CPU plans and retires GPU history
and workspace when the resolution changes. The per-axis limit is 16384 pixels;
practical limits also depend on available device memory. Legacy captured bundles
remain limited to their saved resolutions.

Super Resolution selects its native inference size from the quality and aspect
controls rather than snapping to a preset canvas. To test an additional size:

```powershell
python tools/prepare_native_fixture.py ./build/custom_input --width 1366 --height 768
& ./build/validation_tools/native_inference_probe.exe ./build/native_release ./build/custom_input ./build/custom_result ./build/weight_source.json
& ./build/validation_tools/direct_memory_probe.exe ./build/native_release ./build/custom_input ./build/weight_source.json ./build/resize_result --resize-sweep
```

The resize sweep exercises the addon adapter on one device/queue, checks resource
retirement and verifies that returning to the first resolution reproduces its
output sequence. Add `--super-resolution` to exercise downscaling and upscaling
during resizing. Extra-size smoke tests check finite output and D3D12 errors;
only the saved regression extents have checked-in output hashes.

After both validations pass, embed the packages:

```powershell
python tools/build_addon.py addon_embedded --ngx-sdk C:\SDK\NGX\include --production-only --embed-runtime C:\Packages\ValidatedRunners --embed-native C:\Packages\ValidatedNative
```

For the example above use `./build/runners_package` and `./build/native_release`
as the embedding paths. Check the final addon resources in a fresh explicit cache:

```powershell
& ./build/validation_tools/embedded_runtime_tests.exe ./build/addon_embedded/phoenix.addon64 ./build/resource_cache ./build/resource_test.json
```

Portable CUDA automatically selects direct FP8 matrix instructions for larger
reductions whose activations are already proven to lie on the model's FP8 grid.
Short reductions and unquantized boundaries retain FP16/ordered execution. The
selected GPU and compiled virtual architecture must both support FP8; older
architecture builds retain the previous implementation. This changes execution,
not the model weights or requested network layers.

For a controlled regression comparison, set `"cuda_direct_fp8": false` in the
portable runner configuration. `"cuda_fp8_columns"` accepts `32` (the measured
default), `64`, or `128`; wider tiles are experimental tuning overrides, not
quality settings. The offline graph profiler reads these same options. Runtime
status reports `validation_ms` separately from preparation, model/transfer, and
resolve times; full frame calls also include transport and synchronization.

`tools/embed_runtime.py` requires validation manifests and embeds runner libraries and kernels. The NVIDIA DLSS5 model DLL remains separate. No NVIDIA DLSS5 DLL or extracted model is included in the repository.

The addon builder writes `report.json` with separate compilation, contract-test
and artifact-verification results and a `failed_stage` on failure. If linking
finishes but the addon disappears or cannot be read, inspect that receipt, the
build logs and security history. Do not treat compiler completion as a successful
package: the final addon must be readable and pass the embedded-resource test.

The game also needs compatible ReShade with addon support and a `Phoenix.json` configuration. Without configuration the panel can appear while inference remains inactive. The addon supports selecting a compatible NVIDIA DLSS5 DLL or a local model folder; model selection does not replace the runner/kernel package.

Normal gameplay does not require a readiness marker and performs no marker-file checks. Processing uses the existing configuration, enable controls and runtime initialization checks. For externally controlled tests, set `"require_gpu_ready_file": true` in `Phoenix.json`; this requires `production_game_phoenix_gpu.ready` in the configured sandbox directory, and removing that marker pauses processing. Diagnostic builds retain marker control by default; set the option to `false` to disable it explicitly. The marker is only an external test switch, not model validation.

## Local storage and relocation

One current set of addon binaries, native kernels and platform SDKs is retained in `artifacts/current`; retained outputs are not automatically newly validated builds. The Vulkan import definition used by the sibling test bench is `third_party/vulkan-loader.def`. Follow `AGENTS.md`: no automatic build/test history, repeated model copies or unsolicited runtime tests.

## Add-on 0.81 Release Assembly

The add-on release number `0.81` is distinct from the standalone SDK version
`0.8.1-beta`. See [release notes](docs/RELEASE_0.81.md) for package contents.

`tools/validate_portable_bridge.py --build-addons --ngx-sdk <NGX include>`
builds both add-ons and runs focused CPU contracts, retaining the current
embedded helper payload. `tools/package_portable_candidate.py` verifies and
assembles the matching optional runtime files. These commands require the
current retained build/validation inputs; they do not create inference evidence.
Use the full source-build instructions above to create those inputs on a fresh
checkout. The optional native x86 package also needs the pinned external
WOW64Ext source and locally verified Windows/driver bootstrap evidence; a clean
checkout alone cannot certify that private-ABI profile for another machine.

After committing maintained source and reviewing the publication tree:

```powershell
python tools/audit_release.py --git-visible
python tools/prepare_addon_release.py --wow64ext <WOW64Ext source> --ref <publication ref>
python tools/publish_addon_release.py --ref <publication ref>
```

Preparation writes one final asset set under `artifacts/current/release-0.81`.
It refuses existing output, changed compile inputs or a publication ref with a
different tree, inspects embedded resources, includes dependency licenses/source,
and checks archive contents/hashes. No models, games, GPU calls or downloads are
needed for this packaging step. The publisher's default is a read-only preflight;
`--publish` explicitly pushes a clean public-identity snapshot/tag and creates
a release. It uses existing Git Credential Manager authentication and verifies
public downloads without writing credentials to files or logs.

## Maintained source contracts

Run `python tools/generate_weight_layouts.py --check` and
`python tools/verify_source_provenance.py` before rebuilding maintained sources.
The first verifies both Original and Lightweight FP8 declarations; the second checks the
compatibility helper's current source inputs separately from its declared import
hashes. Compatibility builds also run that input check automatically. See
[source maintenance](docs/SOURCE_MAINTENANCE.md) for regeneration, intentional
source updates, numerical boundaries, and focused model-free tests.
