# Phoenix DLSS5 1.1.5

- Corrected appearance blending above intensity 1 on native RTX, native CUDA and portable processing paths. The display blend saturates instead of extrapolating shading.
- Synchronized artistic style and overall intensity between Phoenix and NVIDIA outputs. Existing configuration keys remain compatible.
- Replaced short option dropdowns with equal-sized, one-click buttons that wrap on narrow panels.
- Added a rebindable F6 processing toggle, preserving the selected output, model and controls. Hotkeys now have their own tab.
- Replaced the empty capture-latency tile with game FPS. Added an FPS history chart and FPS columns to retained timing comparisons, separated by workload and process.
- Moved appearance controls into Play and flattened nested Image/history sections. Clarified when engine motion and depth guides are unavailable.
- Fixed control-response publication failures: temporary file locks retry without blocking inference or replaying a command. Control-channel errors remain separate from rendering status.

The Windows archives include architecture-matched add-ons and model-free runtime companions. Linux packages retain the Vulkan layer and CPU, CUDA and Vulkan providers. Compatible model weights remain a separate input.

[Validation and measurement scope](VALIDATION_1.1.5.md). Some scenes can still show temporal variation, especially with strong FP8 settings; this release does not claim a general flicker fix.
