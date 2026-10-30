# Phoenix DLSS5 1.0 Beta

**Native RTX is the main event. The portability side quest gets better too.**

- **Quiet update notice:** a clickable label at the top of both addon dashboards opens GitHub when a newer published release is available. Beta tags are included; offline checks stay silent.
- Original and Lightweight models, Phoenix DLSS5 / NVIDIA DLSS5 selection, same-frame comparisons and live image controls.
- Matching x86/x64 ReShade addons, Automatic execution selection and the retained x64 graphics helper. Experimental same-process x86 routes keep their prerequisite checks and helper fallback.
- Native instruction choices follow the actual GPU: compatible older instruction styles are selectable on newer cards, while Automatic prefers native kernels.
- Further native tuning: cached weight-side decoding, aligned tile transfers, deferred Ada copy waits, vector reads and a targeted compatibility register budget. Both models preserve matching-control outputs in the documented isolated studies.
- Faster CPU operand reuse and shared host input preparation; grouped Vulkan weight reads improve the scalar fallback. Default cooperative-matrix paths have different costs. Unhelpful CUDA and boundary-register experiments were rejected.
- Updated native, memory, instruction-style and resolution-scaling evidence. The scaling investigation separates actual pixel growth from padding/token steps.
- A short front-page README and focused installation, controls, compatibility, performance and developer guides.

## Downloads

| Asset | Contents |
| --- | --- |
| `DLSS5-Phoenix-1.0-beta-Windows.zip` | Main x86/x64 addons and graphics helper; native RTX 50 kernel payload |
| `DLSS5-Phoenix-1.0-beta-RTX-Compatibility-Windows.zip` | Experimental x86/x64 package with RTX 20/30/40/50 images and selectable instruction styles |
| `DLSS5-Phoenix-1.0-beta-Input-Windows.zip` | Optional Phoenix Input companion for both architectures |
| `DLSS5-Phoenix-1.0-beta-Portable-Runtimes.zip` | Optional x64/x86 runtime companions; available compiled backends and their support limits are documented separately |
| `DLSS5-Phoenix-1.0-beta-Source.zip` | Audited source, tests, documentation and notices |

Choose **one** main Windows package. Close the game, replace the addon matching its process architecture and the bundled graphics helper, and keep your configuration and separately obtained compatible model. Use [ReShade with full addon support](https://reshade.me/). Phoenix Input is separate and optional; rendering does not require it.

## Validation and limits

This is a **beta prerelease**, not a claim of universal game or GPU compatibility. Fresh build, CPU, software D3D12, DLL lifetime, extraction and package checks are recorded in [release validation](VALIDATION_1.0-beta.md) and the downloadable `VALIDATION.json`.

The native kernel images come from the authenticated model-free bundle used by the earlier isolated RTX 5080 studies, including both models and resolutions through 8K. Those measurements retain their dates, fixtures and source identities; they are not newly run game benchmarks of these ZIPs. No retail games are launched for this release procedure.

I do not own RTX 20/30/40 cards. Their kernels compile and pass the documented offline checks, but physical-device inference, quality and performance remain experimental. Metal, HIP and SYCL source support is not a new binary or device qualification in this release. See [compatibility](COMPATIBILITY.md) and [native architecture limits](NATIVE_ARCHITECTURES.md).

No NVIDIA model DLLs, weights, private configuration, captures or diagnostic logs are bundled. ZIPs include licenses and file inventories; `SHA256SUMS.txt` pins the published downloads. NVIDIA's models and external runtimes retain their own terms.

[Installation](INSTALLATION.md) · [User guide](USER_GUIDE.md) · [Performance](PERFORMANCE.md) · [Documentation index](README.md) · [Upstream thanks](UPSTREAM_CREDITS.md)
