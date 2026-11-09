# Phoenix for native Linux — experimental preview

This frontend is a per-process Vulkan layer, independent of ReShade. It is
intended to provide a reusable capture route for native Linux Vulkan games,
starting with 0 A.D. Gameplay with AMD rendering and RTX 5080 inference is recorded in
[the 0 A.D. validation report](LINUX_0AD_VALIDATION.md). Compilation and software
contract tests alone do not establish GPU or live-game behavior. Direct RTX
presentation, Proton and broader game compatibility remain unvalidated.

## What this preview does

Captures an SDR RGBA8/BGRA8 swapchain image, waits for readback, calls Phoenix's
bounded host-plane C API, uploads the result and presents it. The selected
inference backend can run on Vulkan or optional CUDA paths; image transport still crosses the CPU.
This is a synchronous correctness baseline, with a potentially substantial
frame-time cost. The CPU backend is a reference route, not a real-time target.

The HUD is included. Engine depth/motion/UI separation are unavailable. Every
frame resets model history rather than inventing temporal guides. Encoded SDR
channel values are preserved across capture; alpha is retained. HDR, protected
content, array swapchains, presentation extension chains, swapchain extension
chains and batched presentations pass through. A graphics-capable presentation
queue is required. Separate rendering/presentation families require the game
to transfer ownership correctly, as with ordinary presentation. This preview
does not add ownership transfers for the game.

The current portable Vulkan provider supports one active session per process.
Extra live swapchains may pass through if their session cannot be initialized.
Normal swapchain replacement retires the old session and initializes a fresh
one. Initialization failure passes frames through with a diagnostic. Inference
failure retains original pixels. Once game semaphores have been consumed,
Vulkan submission/device failures are returned instead of incorrectly waiting
on the same binary semaphores again.

## Use a complete package

Requires desktop Linux, the package's CPU architecture, Python 3.10+, a Vulkan
loader **1.3.234+** (for additive layer discovery), and the system C/C++ runtime.
The Vulkan compute backend additionally requires the capabilities listed in
[LIBRARY.md](LIBRARY.md). Game Vulkan support alone does not establish compute
backend support. Models are external and retain their own terms.

1. Extract the model-free `phoenix-linux-<architecture>.tar.gz` to a directory
   you own. It can be moved later. No root install or global layer registration
   is required.
2. Run `./bin/phoenix --doctor` there. It verifies packaged file checksums and
   reports included backends and limitations. It does not execute GPU work.
3. Create a runtime JSON file outside the package. Example Vulkan config:

   ```json
   {
     "backend": "vulkan",
     "device": "vulkan:REPLACE_WITH_32_HEXADECIMAL_UUID_DIGITS",
     "allow_unvalidated": true,
     "weight_source": {
       "kind": "model_folder",
       "model": "small-fp8",
       "path": "/absolute/path/to/compatible/model"
     }
   }
   ```

   List compute candidates with `./bin/phoenix_frame --vulkan-devices "$PWD/lib"`
   from the package directory, or use your GPU's `deviceUUID` from Vulkan device
   information (remove dashes). Device enumeration does not validate inference.
   `allow_unvalidated` explicitly opts into an unvalidated device; the launcher
   does not add it for you. Consult [model selection](MULTI_MODEL.md) for model
   source layouts. The launcher supplies `runtime_root` from its package path.
4. Check the config paths with
   `./bin/phoenix --doctor --config /absolute/path/phoenix.json`.
5. Launch a Vulkan game:

   ```sh
   /path/to/package/bin/phoenix --config /path/to/phoenix.json -- 0ad -conf=rendererbackend:vulkan
   ```

   Renderer arguments depend on the game. For a native Linux Steam game, the
   launch option has the form:

   ```text
   /path/to/package/bin/phoenix --config /path/to/phoenix.json -- %command%
   ```

Steam runtimes, Flatpak and other sandboxes must be able to see the package,
configuration, model and system libraries. Proton may require additional layer
discovery and ABI work; the native launcher alone is not verified Proton support.
OpenGL games need a future GL frontend or a separately validated Zink route.

Only the launched process tree receives the layer environment. Existing layer
settings are preserved. Remove the launcher from the game's launch command to
disable Phoenix. To uninstall, remove the extracted package directory and any
configuration you created. No global files or game binaries were modified.

## Native CUDA on RTX 50-series

Packages built with the native CUDA option can run the production Windows native
model schedule through a Linux CUDA executor. See [the native CUDA guide](LINUX_NATIVE_CUDA.md)
for the `native-cuda` configuration, reset-only contract, GPU/API timing results,
and build instructions. The portable `cuda` backend is a separate optional path.
These payloads require the NVIDIA driver only when selected; Vulkan remains
available for other supported GPUs. Models remain external.

## Select game and inference GPUs separately

`--game-device vulkan:<UUID>` restricts the game's Vulkan device enumeration to
one GPU. The runtime JSON's `device` independently selects Phoenix's inference
GPU. For example, a desktop driven by integrated graphics can keep the game on
that GPU and run inference on a discrete RTX card:

```sh
/path/to/package/bin/phoenix \
  --game-device vulkan:REPLACE_WITH_DISPLAY_GPU_UUID \
  --config /path/to/phoenix-rtx.json -- 0ad -conf=rendererbackend:vulkan
```

Use the complete `vulkan:` ID from device discovery; both IDs must be available
to the Vulkan loader. In particular, a process-wide `VK_DRIVER_FILES`,
`VK_ICD_FILENAMES` or device-selection filter that hides the RTX also hides it
from inference. Remove those restrictions from this launch's environment when
using separate GPUs. The launcher preserves the caller's environment settings;
it does not silently remove driver filters.

The opt-in game filter requires an application Vulkan instance version of at
least 1.1 for core UUID queries. It filters physical-device and device-group
enumeration consistently. An unavailable UUID produces an explicit error,
without substituting a different GPU. Without `--game-device`, enumeration
behavior is unchanged. `PHOENIX_GAME_DEVICE` is the equivalent environment
setting for custom launchers.

Phoenix scopes its own runtime calls separately: Vulkan instances created by
those calls retain the available device list. The process-wide environment is
not changed while initializing inference, and concurrent game threads retain
their own application visibility. The current synchronous host-buffer transport
also moves data between the two GPUs through CPU memory. This option does not
provide GPU-to-GPU zero-copy transport or repair a broken display-driver route.

## Build and package

The layer itself needs ordinary desktop Vulkan headers. Full Vulkan inference
also needs the newer pinned headers/compiler documented in [LIBRARY.md](LIBRARY.md).
Build in a temporary directory and remove it after promoting current outputs.

```sh
cmake -S native_runtime -B /tmp/phoenix-linux-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPHOENIX_BUILD_LINUX_LAYER=ON \
  -DPHOENIX_ENABLE_VULKAN=ON \
  -DPHOENIX_VULKAN_HEADERS=/path/to/Vulkan-Headers/include \
  -DPHOENIX_GLSLANG=/path/to/glslang
cmake --build /tmp/phoenix-linux-build --parallel 4
ctest --test-dir /tmp/phoenix-linux-build --output-on-failure -R '^linux_'
cmake --install /tmp/phoenix-linux-build --prefix /tmp/phoenix-linux-stage
python3 tools/package_linux.py --prefix /tmp/phoenix-linux-stage \
  --output artifacts/current/linux --version 1.0.1-linux-preview
```

The packager requires the Vulkan provider unless `--reference-only` is explicitly
selected. Reference-only archives are named accordingly and cannot select Vulkan
inference. It bundles the launcher, layer, runtime, backend/shaders, CLI,
documentation and applicable third-party notices. It records file hashes, source
revision/dirty state, CPU architecture, minimum runtime requirements and untested
features. It does not bundle proprietary weights, Windows DLLs, the system Vulkan
loader, GPU drivers or compiler toolchains. The archive and SHA-256 sidecar are
the current deliverables. Checksums detect accidental changes, not authenticity;
release signing remains a publication task.

Packaging needs `readelf` from binutils. The manifest records required GLIBC,
GLIBCXX and CXXABI symbol versions from the actual ELF files, plus their shared
library dependencies. Building on a newer distribution can raise these minimums;
use the oldest supported distribution for public release builds. CI uses Ubuntu
22.04; local archives retain the requirements of the local compiler and system.

## Validation and next steps

Software tests cover encoded-pixel round trips, alpha preservation, bounded
conversion, launcher environment/configuration isolation and package integrity.
These are not live-game or GPU tests. Before describing a release as game-ready,
validate the packaged layer on actual Linux Vulkan games with the validation
layer: resize, fullscreen changes, focus, shutdown, model failures, multiple
queues and repeated swapchain replacement. Measure total frame time, not just
inference. Record the exact GPU/driver, package hashes and game versions.

Next development steps are asynchronous processing with correct frame pairing,
GPU-resident image transport, optional motion estimation, per-game UI protection,
presentation-extension support, and independently validated Proton/OpenGL routes.
No native RTX Windows benchmark is a Linux throughput claim.

Set `PHOENIX_DIAGNOSTICS=1` for optional input/output byte-hash comparison.
On normal swapchain retirement or shutdown, the layer logs captured, successfully
processed, failed and presented frame counts, changed-frame counts when hashing
is enabled, and mean/first host-call inference time. These timings include the
runtime's host work and internal transfers, not the complete game frame. Hashing
adds diagnostic overhead. No frame pixels are written to disk by this option.

## Architectural reference

[vkBasalt](https://github.com/DadSchoorse/vkBasalt), revision
`4f97f09ffe91900e6ca136cc26cf7966f8f6970d`, was cloned separately and consulted for
Linux Vulkan-layer architecture. This Phoenix layer is independently implemented;
vkBasalt source and its ReShade compiler are not copied or required. vkBasalt's
zlib license remains with the separate checkout. Vulkan loader chaining follows
the [Khronos loader interface](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderLayerInterface.md).

Set `PHOENIX_PROFILE=1` for bounded stage timing: Vulkan readback, input preparation,
inference, output packing, upload, present API call, total hook and presentation
interval. Summaries use at most the last 256 successful samples after skipping
three initial captured frames per swapchain. `present_fps` is 1000 divided by the
mean interval between hooks; it includes game work/pacing. Keep byte-hash
diagnostics and Vulkan validation off for performance runs, and validate
correctness separately. Cached coherent readback memory is preferred when
available, with the existing coherent fallback retained.
