# Phoenix DLSS5 1.3.3

- **Compact status and timings.** API, game architecture, processing route, model, resolution, and GPU details share one status box. A shared render and inference GPU appears once. Game FPS, fresh DLSS5 FPS, neural time, and pipeline round trip use four compact cards.
- **Render and inference device identification.** Same-device processing, the staged worker, and the external native helper report their processing device. Split-GPU setups can show the integrated game GPU alongside the RTX inference GPU.
- **Actionable wait messages.** Sustained capture/readback, inference, and output-return waits appear in the status box. An older inference submission takes priority over newer captures waiting behind it. Disabled processing and failed routes suppress stale wait messages.
- **Responsive layout.** GPU details wrap at narrow widths, and status information is available across the overlay tabs.

DLSS5 Everything can install and verify the matching addon and runtime components. See the [installation guide](INSTALLATION.md) and [validation details](VALIDATION_1.3.3.md).
