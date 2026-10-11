# Route controls and resolution bridge

- NVIDIA DLL output uses the game's rendering GPU. Phoenix model and runner controls apply only to Phoenix output or the Phoenix side of a comparison.
- The shared **Resolution mode** offers **Normal** and **Super resolution**. Normal retains the original inference dimensions and copy path. Super resolution reduces the image before NR inference and returns it to the original output dimensions.
- Processing quality selects a pixel budget. Processing aspect offers game aspect, 16:9, 16:10, 21:9, 4:3 and a custom ratio. Padding preserves the original image aspect; the native runner selects an existing validated canvas. Actual canvas, active image and output dimensions are displayed. Unsupported smaller native canvases fall back visibly to Normal.
- The shared **Upscaler** choice offers spatial reconstruction, NVIDIA DLSS Super Resolution and Intel XeSS for both Phoenix and NVIDIA NR output. NR inference and upscaling are separate operations. Temporal upscaling runs on the rendering GPU; it needs observed motion/depth and a compatible separately supplied runtime. The DLL route does not gain a Phoenix runner selector.
- DLSS SR uses the game's initialized NGX context without reinitializing or shutting it down. XeSS queries acceptable input resolutions before selecting a vendor quality mode. Both receive active-image subrects, scaled motion and jitter; padding is excluded. Missing guides/runtime or unsupported initialization produces a visible spatial fallback. A vendor evaluation failure follows the existing failed-generation guard rather than submitting uncertain resource states.
- Normal mode does not instantiate a temporal upscaler. Temporal output adds one full-resolution RGBA16F owned surface, accounted as bridge memory. Vendor internal allocations are opaque and excluded from the owned ledger; process GPU-local usage includes them but also harness/driver resources.
- Motion vectors, jitter and depth are mapped to the active image. Upscaling excludes padding and preserves the original output alpha. Engine-buffer descriptors remain immutable while commands are in flight. Both inference and bridge-only submissions fence their resources before replacement.
- RenoDX-specific widgets were removed. The internal foreign-addon filename check remains to prevent double processing; third-party notices are preserved.
- Timing measurements live outside replaceable GPU recorders. Rows retain their route, device, resolution and settings context across switches. Current summaries filter that context. Rows also retain the actual upscaler status, so spatial fallback samples do not merge into active vendor samples. Clearing timings invalidates pending old samples too. The archive is bounded to 128 rows and a 120-sample plot.

## Initial validation

The source build passed 11 CPU suites and two D3D12 WARP suites. The resampling suite checks a gradient numerically, exact motion scaling and padding, alpha preservation and two rotating engine buffers recorded before submission, with the D3D12 debug layer enabled. DLL-only bridge construction is tested without a Phoenix model or worker.

The native RTX direct-adapter probe at 1920×1080 matched all 48 Normal-mode output hashes from the previous build across 1, 2 and 4 refinement passes and reconstruction after teardown. A 1280×720 inference canvas returned finite 1920×1080 output with original alpha intact. Both probes reported zero D3D12 errors and zero owned allocations after teardown. These checks establish execution and regression behavior, not temporal visual quality in gameplay.

Local evidence is regenerated under `build/resolution_final_probes/native_normal`, `native_spatial`, `native_dlss` and `native_xess`. The final embedded addon is in `build/resolution_temporal_release`.

## Temporal upscaler validation

On NVIDIA GeForce RTX 5080, with NVIDIA DLSS SR runtime 310.3.0.0 and XeSS runtime 2.0.2.68, both vendor APIs passed 12 stationary synthetic frames, including a mid-sequence reset, at 1280×720 → 1920×1080 and padded 640×384 (active 640×360) → 1920×1080. Each case also exercised six frames through the addon's DLL-only encode/guide/resolve/upscale/submission path with a synthetic stand-in NR surface. All output pixels were finite, output was nonconstant, original game alpha was preserved, D3D12 reported zero errors, and all runner-owned objects retired after the fence.

Actual native Phoenix inference followed by each vendor upscaler also passed 12 captured-color/motion frames at 1080p, using synthetic constant depth. All owned allocation bytes and objects returned to zero. These are execution, composition and lifecycle checks; NVIDIA NR inference is not executed by the synthetic DLL-only probe. Gameplay quality, Intel hardware and dual-GPU temporal quality remain unverified.

The optional offline DLSS harness links the official external NGX SDK initialization library. The addon requires no additional NGX link library and reuses the game's context. XeSS headers and notices are tracked; vendor binaries and all models remain external. See [dependency instructions](../third_party/EXTERNAL_DEPENDENCIES.md).

The compact receipt is `build/resolution_final_probes/validation_summary.json`.
The embedded-payload check verified all 118 files and cache reopening, including
a long cache path that previously exceeded Win32's conventional path limit.
The loader now normalizes extended local/UNC paths for its explicit file calls.
