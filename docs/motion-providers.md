# Windows motion providers

Choose the estimator in **Image → Motion and scene changes → Motion source**.
Automatic uses engine vectors when available, then FidelityFX. If FidelityFX
is unavailable, it tries NVIDIA Optical Flow. OpenCV DIS and D3D12 Video are
explicit selections, as is Luma blocks. Missing drivers or payloads leave spatial
processing active and show the reason in motion status.

- **FidelityFX** runs on the Direct3D 12 compute pipeline. Its SDK suppresses
  motion for six frames after a reset. It needs the x64 provider payload.
- **OpenCV DIS** runs UltraFast on the CPU. Phoenix downsamples on the GPU,
  reads grayscale back, and uploads the resulting dense flow. It needs the
  x64 provider payload.
- **D3D12 Video** uses the driver's NV12 motion search. Analysis dimensions
  round up to even pixels. Available block sizes and extent are driver-specific.
- **Luma blocks** is a built-in HLSL prototype. It compares a sparse luma patch
  per 8×8 block against the previous frame, refines the winning offset at
  one-pixel steps, and rejects weak or ambiguous matches. It needs no provider
  DLL and has no subpixel refinement. Use 360 analysis height as a starting point.

The Windows x64 final-frame route and Native RTX helper use the same motion
contract. An x86 addon uses the x64 helper. Each provider maintains previous-frame
data per stream. NVIDIA flow retains its confidence controls and diagnostic
views. The alternatives reject vectors that reproject outside the frame.
FidelityFX also uses its scene-change history bits when scene detection is enabled.

## Build

Use Visual Studio 2022 with the Windows SDK and CMake. Use FidelityFX SDK 1.1.4
and OpenCV 4.12.0. Point the paths below at separate SDK checkouts.

~~~powershell
cmake -S $ffxSdk/sdk -B build/ffx-motion -G "Visual Studio 17 2022" -A x64 -DFFX_API_BACKEND=DX12_X64 -DFFX_OF=ON -DFFX_BUILD_AS_DLL=OFF
cmake --build build/ffx-motion --config Release --parallel
$opencvSource = (Resolve-Path $opencvSource).Path
$opencvBuild = Join-Path (Get-Location) build/opencv-motion
$pathMap = "/experimental:deterministic /pathmap:$opencvSource=OpenCV /pathmap:$opencvBuild=OpenCVBuild"
cmake -S $opencvSource -B build/opencv-motion "-DCMAKE_CXX_FLAGS=$pathMap" "-DCMAKE_C_FLAGS=$pathMap" -G "Visual Studio 17 2022" -A x64 "-DBUILD_LIST=core,imgproc,video" -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DWITH_IPP=OFF -DWITH_OPENCL=OFF -DWITH_ITT=OFF -DWITH_EIGEN=OFF -DWITH_PROTOBUF=OFF -DWITH_ADE=OFF -DBUILD_WITH_STATIC_CRT=ON -DCPU_DISPATCH=AVX2
cmake --build build/opencv-motion --config Release --parallel
# OpenCV embeds a generated summary with machine-specific paths.
'"OpenCV 4.12.0 (Phoenix DIS provider)"' | Set-Content build/opencv-motion/modules/core/version_string.inc
cmake --build build/opencv-motion --config Release --target opencv_core --parallel
python tools/build_motion_providers.py --fidelityfx-sdk $ffxSdk/sdk --opencv-source $opencvSource --opencv-build build/opencv-motion --output build/motion-payload
~~~

Pass `--motion-providers build/motion-payload` to the existing
`tools/build_current_compatibility.py` command. It embeds both DLLs and their
notices in the x64 and x86 packages. The standalone x64 native runtime resolves
the same optional DLLs beside `phoenix_native.dll`.

## Benchmark

~~~powershell
python tools/benchmark_motion_providers.py --addon32 $addon32 --weight-source $originalConfig $fp8Config --output artifacts/current/motion-benchmark.json
~~~

Each configuration is a weight-source JSON for a locally available model.
The benchmark uses a 2560×1080 fixture with moving foreground and background
textures. It checks the active provider, temporal-guide coverage and changed
neural output. Motion timing covers preparation and waits. Round-trip timing
covers the x86 shared-memory transport, motion, two neural refinements, and
returning the output. Game capture and presentation are outside this benchmark.

The translation fixture in `tests/native/pipeline_motion_accuracy_tests.cpp`
accepts the argument `all` when the provider DLLs are beside the executable.
It measures error in full-frame pixels using a known 8×4 pixel translation
and checks the D3D12 debug queue.

Run the translation fixture with the argument `block-match` to compare wall
motion timing and interior coverage for FidelityFX and Luma blocks at 2560×1080.
