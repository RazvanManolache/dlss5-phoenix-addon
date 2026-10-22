# 0.14 validation scope

This release is CPU/software-rendering-checked only. No hardware GPU inference,
game run, device benchmark or installed-game modification is part of validation.
The downloadable receipt binds fresh checks to the exact shipped add-on hashes.

## Fresh checks

- Rebuild both add-on architectures and the graphics helper, for standard and
  experimental RTX compatibility packages. Core builds run the existing 22 CPU
  suites and two software-D3D12 suites.
- Check x86/x64 routing regressions: NVIDIA DLSS5/comparison helper selection,
  rejection of forced unsafe routes, precise transport refusal reasons and
  terminal failure only after asynchronous discovery is complete.
- Load/unload the real packaged add-ons against a minimal ReShade registration
  host, verify feeder ABI and balanced callbacks, and exercise payload extraction.
  This is not game presentation or manual UI validation.
- Audit nested embedded resources, source and ZIP entries for private paths,
  credentials, device identifiers, logs, weights and NVIDIA model DLLs.
- Check archive inventories/CRCs, committed source identity and public-download
  SHA-256 values. Preserve experimental markers in both add-on architectures.

## Retained evidence

Runtime companions and native kernels must match the hash-verified public 0.13
payloads. Earlier CPU model/fusion checks apply to those unchanged runtime
binaries; no new inference or performance result is claimed for 0.14. The
regular package retains single-target RTX 50 kernels and the compatibility
package retains experimental older-RTX images. No hardware qualification changes.

The withdrawn NVIDIA-in-x86 experiment has no new compatibility claim. Phoenix
DLSS5's eligible in-process x86 routes remain experimental and profile-dependent;
profile-specific bootstrap packages are not shipped. The retained loader
diagnostic changes compile separately without executing NVIDIA code or creating
a hardware graphics device.

Older physical RTX cards, retail-game presentation, GPU-upload fallback,
device-loss/resize behavior and end-to-end NVIDIA DLSS5 execution remain outside
this release's fresh checks. Historical benchmark figures retain their original
dates, binary identities and scope.
