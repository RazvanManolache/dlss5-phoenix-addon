# Phoenix DLSS5 1.0 Beta 2

**Less time shuffling pixels. More ways to keep the renderer inside your game.**

- Automatic routing prefers an eligible in-process renderer, including local D3D11/D3D12 bridges. Unsupported features and resource limits retain helper fallback.
- Experimental x86 library routes gain motion services and compact GPU pixel conversion. Profiles negotiate these capabilities; older companions keep their existing path.
- Missing engine motion can be estimated, with configurable analysis resolution and history/reset controls on supported routes.
- DX9 capture uses a bounded two-slot queue when resources allow. Input/output frame identity remains paired through capture, processing, display and screenshots.
- Native workers can pack and read back output in the inference submission instead of adding a separate output submission and wait. Older SDKs retain their compatible path.
- Experimental x86 bridges discover the installed NVIDIA driver, validate it in an isolated model-free process, and cache the result by driver, package and Windows identity.
- Runtime continuity, comparison switching, worker ownership and shader-compiler lifetime have additional checks and regression coverage.
- Safer runtime recreation and unloading: retiring workers retain their provider callbacks, and background staging no longer keeps a game-runtime pointer alive by assumption.

Both Original and Lightweight FP8 models remain supported. The main README keeps the earlier isolated Phoenix DLSS5 / NVIDIA DLSS5 performance and memory measurements with their original dates and scope. These changes are not a claim of a universal game-FPS improvement.

## Downloads and installation

| Asset | Purpose |
| --- | --- |
| `DLSS5-Phoenix-1.0-beta2-Windows.zip` | Main matching x86/x64 addons and graphics helper |
| `DLSS5-Phoenix-1.0-beta2-RTX-Compatibility-Windows.zip` | Experimental RTX 20/30/40/50 instruction images |
| `DLSS5-Phoenix-1.0-beta2-Input-Windows.zip` | Separate optional input/automation companion |
| `DLSS5-Phoenix-1.0-beta2-Portable-Runtimes.zip` | Optional runtime companions and experimental x86 bridge files |
| `DLSS5-Phoenix-1.0-beta2-Source.zip` | Audited source, tests, documentation and notices |

Close the game and install **one** matching Windows addon package with [full-addon-support ReShade](https://reshade.me/). Replace the matching architecture's addon and graphics helper together. Update optional runtime companions together too; retain your configuration and separately supplied compatible model. Start with Automatic execution. Input hooks are optional and are not needed for rendering.

## Validation and limits

This is a **beta prerelease**. Fresh build, CPU contracts, software D3D12, DLL-load, embedded-payload, source and archive checks are described in [validation](VALIDATION_1.0-beta2.md) and recorded with binary identities in the downloadable `VALIDATION.json`.

The release procedure does not launch games, run hardware-GPU inference or repeat the performance studies. Earlier game and isolated-GPU evidence belongs to the specific builds tested. Inactive-desktop execution is test tooling, not a claim that every game renders while unfocused.

RTX 20/30/40 physical-device qualification remains outstanding; the available test GPU is an RTX 5080. Experimental same-process x86 execution still depends on Windows compatibility, driver validation, available address space and requested features. Automatic helper fallback remains important.

No NVIDIA model DLLs, model weights, private game configuration, test captures or diagnostic logs are distributed. Licenses and per-file hashes accompany the packages. External NVIDIA models and runtimes retain their own terms.

[Installation](INSTALLATION.md) · [Controls](USER_GUIDE.md) · [Execution routes](EXECUTION_MECHANISMS.md) · [Frame delivery](FRAME_DELIVERY_OPTIMIZATION.md) · [Upstream thanks](UPSTREAM_CREDITS.md)
