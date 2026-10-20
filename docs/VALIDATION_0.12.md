# 0.12 Validation Scope

The release receipt identifies the exact source commit and hashes of the shipped
standard and experimental x86/x64 add-ons. Package preparation refuses receipts
that do not match the binaries and audits each nested embedded payload.

- Both Windows variants are rebuilt from maintained source, including the current
  model-picker policy and x86 OpenGL route exclusions. Each core build runs 22 CPU
  contract suites and two software-D3D12 suites.
- Architecture checks independently extract all 164 compiled device images,
  verify their declared architectures and entry points, and compare all 41
  `sm_120` images byte for byte against the retained validated baseline.
- Targeted CPU checks cover model selection, packed arithmetic conversions,
  graph/launch contracts, release channels, packaging guards and privacy exclusions.
- Isolated RTX 5080 checks use both models and payloads extracted from the actual
  rebuilt add-ons, with reset/history readback comparison at 320x180 and 1920x1080.
  The final machine-readable receipt records which cases passed. No retail games
  are launched or modified.
- The standard payload retains its existing single-architecture validation.
  Multi-architecture payloads remain explicitly experimental. No older GPU's
  validation state is changed to validated based on testing an RTX 5080.
- Source and release archives exclude model DLLs/weights, logs, private paths,
  device identifiers and private configuration. Archive contents and public
  downloads are checked against their SHA-256 inventories.

These checks do not establish RTX 20/30/40 physical-device compatibility, retail
game presentation, NVIDIA-backend compatibility through the patched DLL, or
performance parity on older GPUs. The earlier development timings in
[native architecture notes](NATIVE_ARCHITECTURES.md) remain tied to their original
binaries; they are not relabeled as a new 0.12 benchmark.
