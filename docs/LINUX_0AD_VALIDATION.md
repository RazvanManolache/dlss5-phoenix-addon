# Linux 0 A.D. validation and native CUDA performance

On 2026-11-09 the native CUDA executor ran both full model schedules on Linux,
and the Lightweight model completed bounded 0 A.D. gameplay with resize,
camera input and normal shutdown. The package remains an **SM120 reset-only
preview**. Game rendering/presentation used AMD integrated graphics; neural
inference used the RTX 5080. No model data is bundled.

## Native model timing at 1920×1080

The input was the retained 800×450 game-only fixture, resized to full HD with
Pillow's default RGB bicubic resize. Each case used ten initial calls and 24
measured calls, fixed reset input, automatic masking enabled and inverted depth
disabled. GPU clocks were not changed.

| Model | Neural launches | Median model GPU interval | Median synchronous RGBA8 call |
| --- | ---: | ---: | ---: |
| Original | 154 | 3.929 ms | 5.010 ms |
| Lightweight FP8 | 78 | 2.176 ms | 3.229 ms |

The GPU interval includes counter initialization and the complete native
stem/model/head. The synchronous call also includes pinned staging, transfers,
GPU byte conversion and copying the result back to the caller. Neither column
includes game rendering or Vulkan readback/upload. These are not game FPS.
The production planner, authenticated packed weights and kernel sources are
shared with Windows; Linux uses CUDA texture objects and graph dispatch.

## Live-game performance

0 A.D. 0.28.0, Snap 748's native executable, ran `skirmishes/skirmish_demo_2p`
with a Vulkan renderer, windowed mode, vsync disabled and sound disabled.
HOME/XDG profiles were isolated. The desktop used Ubuntu 24.04, kernel
6.14.0-27-generic, Mesa 25.2.8 and NVIDIA 595.99.02.

| Route and extent | Successful frames | Mean inference call | Mean complete hook | Measured presentation rate |
| --- | ---: | ---: | ---: | ---: |
| Earlier portable Vulkan, 800×450 | 429 | 19.790 ms | 60.442 ms | 13.694 FPS |
| Cached coherent readback fix, 800×450 | 752 | 18.744 ms | 30.790 ms | 23.432 FPS |
| Native CUDA RGBA8, 800×450 | 866 | 1.446 ms | 8.716 ms | 62.171 FPS |
| Native CUDA RGBA8, 1920×995 | 1078 | 3.183 ms | 23.431 ms | 40.186 FPS |

Timing means use the last 256 successful samples after three initial frames per
swapchain. Presentation rate is 1000 divided by mean interval between hooks;
it includes game work and pacing. These were independent bounded runs, not a
matched-frame statistical benchmark. The window manager constrained the game's
requested 1920×1080 resize to **1920×995**; the standalone model test above
was actually 1920×1080. All 1,944 frames in the current performance run processed
and presented without inference failures; the launcher exited 0.

The principal fixes were cached coherent Vulkan readback memory (pixel unpacking
fell from 28.060 to 0.271 ms), the production native kernel schedule, and bounded
RGBA8 transport with CUDA conversion. Full-frame RGB32F host conversion is no
longer on the native presentation path. At 1920×995, the remaining readback wait
averaged 17.907 ms and includes queued AMD game rendering, versus a 3.183 ms
inference call. This is still synchronous host transport, not zero-copy.

## Correctness evidence and limits

A separate run enabled input/output byte hashing and the actually loaded
`VK_LAYER_KHRONOS_validation` **1.3.275**, verified in the game process mappings.
Synchronization validation was requested. It processed, changed and presented
**1,383/1,383 frames**, with zero inference failures, zero captured Vulkan
validation errors/warnings, zero synchronization-hazard reports, resize/camera
input exercised and launcher exit 0. This validator checks the AMD game and
presentation frontend; the CUDA executor does not use Vulkan compute features.

CUDA **compute-sanitizer 13.1.75 memcheck** separately reported **zero errors**
across 16 native runtime contract checks. Checks covered repeated reset,
nonfinite/capacity/metadata rejection without output writes, recovery after
validation errors, input/output aliasing, output row padding, and exact RGBA8/
BGRA8 results against the float V2 route with arbitrary alpha preserved.
Zeroing the temporary diagnostic executor's uploaded weights changed its output;
the external model DLL was not changed. Both default D3D12 boundary cubins also
remained byte-exact against HEAD under Linux CUDA 13.1.80 device compilation.
This last check is not a Windows runtime test.

Both native full-HD cases produced finite, repeatable results. The RGBA8 hashes
were equal on this one reset fixture; this is not a model-distinction or quality
test. **No new Windows/NVIDIA pixel-parity test was run.** Changed pixels, finite
output and a visually plausible game capture do not certify general fidelity.
The frontend includes the HUD and resets history every frame. Long sessions,
fullscreen, device loss, other games and Proton remain untested. Direct NVIDIA
presentation still fails on this desktop; the opt-in AMD game-device filter
avoids that route while CUDA independently selects the RTX.

Compact current receipts are `PERFORMANCE.json`, `CUDA_VALIDATION.json`,
`0AD_VALIDATION.json` and `VALIDATION.json` in `artifacts/current/linux`.
`0ad-gameplay.png` is the earlier fixed benchmark input; `0ad-native-gameplay.png`
is the current game-only performance-run capture. The current model-free archive
is independently checksummed. See [native CUDA setup](LINUX_NATIVE_CUDA.md) for
configuration, the bounded API and reproducible build/test commands.

The portable run below predates this executor and used a different validator
and binary. Its timings/warnings are historical, not current native results.

## Earlier portable Vulkan validation

The native Linux frontend passed a bounded **split-GPU gameplay smoke test** on
2026-11-09: **0 A.D. rendered and presented on AMD integrated graphics while
Phoenix Lightweight FP8 ran on the RTX 5080**. The opt-in `--game-device` filter
avoids the failing NVIDIA presentation-mode enumeration without changing display
or driver settings. This is functional evidence for this configuration, not a
general game-compatibility or visual-parity certification.

### Tested configuration

- Game: **0 A.D. 0.28.0**, Snap revision **748**. The installed native
  `binaries/system/pyrogenesis` executable was run directly with its supplied
  shared libraries. The confined Snap launcher was not validated.
- OS: Ubuntu 24.04, kernel `6.14.0-27-generic`, X11 desktop.
- Game GPU: AMD Ryzen 9 9950X3D integrated graphics, RADV
  `RAPHAEL_MENDOCINO`, Mesa **25.2.8**;
  `vulkan:fedcba9876543210fedcba9876543210`.
- Inference GPU: **NVIDIA GeForce RTX 5080**, driver **595.99.02**;
  `vulkan:0123456789abcdef0123456789abcdef`.
- Runtime: portable **vulkan-cpp** backend, reference arithmetic, cooperative
  matrix acceleration enabled. This is separate from Windows Native RTX/D3D12.
- Model: **small-fp8**, read as resources from the pinned compatibility DLL
  **310.9.16.0**. The Windows DLL was not executed. DLL SHA-256:
  `cdfa6c1ec784867182164c7836f324f087b5c9f374d0c9fe46f54999feabf3ed`.
  The runtime reported resource SHA-256:
  `f743e8eb320f42bc9bdc992ee2bf51f7bb7c8688ced4ecf5ce6807ec4eef7962`.
- Model provenance: DLSS5 Everything's `PayloadBootstrapper.Nvidia.cs`,
  revision `98df8f7a917810bbd48f5aee4eefd8279980986c`, pins NODIX
  `DLSS-5-MANAGER` **1.3.7**. The archive and selected DLL size/hash were
  checked; only the selected DLL was extracted. Model data stays external.
- Validation: locally built `VK_LAYER_KHRONOS_validation` **1.4.365**,
  revision `0afe79ff768afd7243b6f697b07b44ee3b2f59ea`, with
  `khronos_validation.validate_sync = true`. The validator manifest used an
  absolute library path, and `/proc/<game-pid>/maps` confirmed that this exact
  build was loaded. GPU-assisted validation was not enabled.
- Game profile, cache, replay data and logs were isolated in a temporary task
  directory. Installed game files and existing game settings were preserved.

### Observed result

The `skirmishes/skirmish_demo_2p` map reached
`GAME STARTED, ALL INIT COMPLETE`. The 42-second bounded run resized the window
from 640×360 to 800×450 and sent right-arrow camera input. The launcher exited
**0** after requested shutdown. A game-only compositor pixmap capture showed
rendered gameplay at 800×450, without capturing the desktop.

| Swapchain extent | Captured | Processed | Changed | Presented | Inference failures | Mean inference host call | First inference host call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 640×360 | 52 | 52 | 52 | 52 | 0 | 34.254 ms | 279.056 ms |
| 800×450 | 47 | 47 | 47 | 47 | 0 | 52.443 ms | 292.298 ms |

All **99** processed frames changed their encoded pixel-byte hash and were
presented. Runtime status identified the RTX UUID for both swapchains. A sample
from `nvidia-smi` independently showed the same game process using **239 MiB**
of NVIDIA GPU memory; this is a sampled process allocation, not a peak or an
isolated model memory requirement.

The captured successful run had **zero validation errors**, **zero synchronization
hazard reports**, and **zero game error messages**. There were **10 validation
warnings**, all `Undefined-Value-ShaderOutputNotConsumed` from the game's graphics
pipelines writing a fragment output without a corresponding color attachment.
Phoenix's frontend records transfer commands and its runtime uses compute
pipelines; these warnings were not suppressed or repaired by this test.

The capture interval includes loading and gameplay frames. Timings are wall-clock
duration of the synchronous Phoenix C API call, including runtime host work and
internal transfers, and include the first call. They exclude some layer
readback/upload, hashing, validation overhead outside the call, and game work.
They are **not end-to-end FPS or isolated GPU kernel timings**. No NVIDIA visual
parity claim follows from changed pixels or visual inspection. Long sessions,
device loss, fullscreen switching and other games remain untested.

### Earlier AMD-only baseline

Before the split-GPU filter, a bounded AMD-only test completed 25 processed,
changed and presented frames, with zero inference failures or validation errors
under the older system validator **1.3.275**:

| Extent | Frames | Mean inference host call | First call |
| --- | ---: | ---: | ---: |
| 640×360 | 12 | 840.671 ms | 1055.664 ms |
| 800×450 | 13 | 1354.589 ms | 1590.875 ms |

This earlier observation shows the size of the improvement available by moving
inference off this small iGPU; the separate runs and validator versions are not
a controlled performance benchmark. The AMD runtime warned that it could not
declare preservation of float16 subnormals; cross-driver numerical parity was
not established.

### Presentation failure and workaround

Without the game-specific visibility filter, 0 A.D. enumerates every physical
GPU before choosing one. Querying presentation modes on this desktop's NVIDIA
route returns:

```text
vkGetPhysicalDeviceSurfacePresentModesKHR ... VK_ERROR_UNKNOWN
DeviceSelection.cpp:262 (MakeAvailablePhysicalDevice)
```

System `vulkaninfo --summary` independently reproduced this surface-mode error.
A game device-index override alone did not avoid enumeration. A process-wide
AMD-only driver filter allowed the game to run but also hid the RTX from inference.

`--game-device` solves this route by restricting the game instance's device
list to AMD while runtime-owned instances retain the available GPUs. Both GPU
drivers stay visible to the loader; no process-wide environment changes are
made during runtime initialization. Direct NVIDIA presentation remains untested
beyond the failing probe; this workaround does not repair that display route.

The old system validator **1.3.275** also reported unrecognized NVIDIA feature
structures and cooperative-matrix shader capabilities during initial split-GPU
trials. Those diagnostics were resolved by using the actual **1.4.365** library;
no shader feature or validation check was disabled to make the final run pass.

### Reproduction outline

Use external runtime JSON selecting the authenticated Lightweight FP8 resource
and RTX inference UUID, following [LINUX.md](LINUX.md). Preserve Xauthority and
use isolated HOME/XDG directories for a disposable test. Remove process-wide
driver/device filters that would hide either GPU. With a current validator
available, the tested launch has this form:

```sh
VK_LAYER_PATH=/path/to/current-validator/explicit_layer.d \
VK_LAYER_SETTINGS_PATH=/path/to/validation-settings \
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
PHOENIX_DIAGNOSTICS=1 \
LD_LIBRARY_PATH=/snap/0ad/current/binaries/system:/snap/0ad/current/usr/lib/x86_64-linux-gnu \
/path/to/package/bin/phoenix \
  --game-device vulkan:fedcba9876543210fedcba9876543210 \
  --config /path/to/phoenix-rtx.json -- \
  /snap/0ad/current/binaries/system/pyrogenesis \
  -conf=rendererbackend:vulkan -conf=windowed:true \
  -conf=xres:640 -conf=yres:360 -conf=gui.splashscreen.enable:false \
  -nosound -autostart=skirmishes/skirmish_demo_2p
```

The settings directory contains `vk_layer_settings.txt` with
`khronos_validation.validate_sync = true`. Check the validator manifest's
`library_path`: metadata advertising a new version can still resolve an older
system library if the manifest uses only a basename. The local validation build
used `BUILD_TESTS=OFF`, `BUILD_WSI_WAYLAND_SUPPORT=OFF`, `UPDATE_DEPS=ON` and its
pinned dependency revisions. No validator or proprietary model is bundled with
the game package. Test-specific paths and credentials are excluded.
