# 0.13 validation scope

Fresh validation is deliberately CPU/software-only. No hardware GPU inference,
retail game, device benchmark or installed-game modification is authorized for
this release. The downloadable receipt binds checks to exact shipped hashes.

Release checks cover compilation of both add-on architectures and the helper,
CPU contracts, software D3D12 tests, actual add-on attach/detach, embedded payload
extraction, recursive content/hash audits, updated runtime CPU inference,
streamed weights and packaging guards. Runtime dependencies carried forward from
the preceding public archive are hash-verified rather than described as rebuilt.
Current source is rebuilt wherever the model/runtime/CLI implementation changed.

The regular package retains validated single-target RTX 50 kernel images. The
compatibility package retains explicitly experimental RTX 20/30/40 images alongside
the unchanged RTX 50 images. This release does not promote any hardware status.
The archived [pipeline measurements](benchmarks/current/pipeline/README.md) and
[main performance report](benchmarks/current/README.md) are earlier observations,
not a new measurement of these packages.

CPU model checks do not validate NVIDIA DLSS5 execution, GPU-upload fallback,
physical older cards, retail-game presentation, resizing/device loss or manual
UI behavior. New chain services remain route-gated; experimental x86 native
bridges reject unsupported services rather than silently skipping them.

Source, nested embedded resources and final ZIP entries are screened for private
paths, device IDs, credentials, logs, model weights and NVIDIA DLLs. Source ZIP
contents must match the audited committed tree; binary inventories and public
download checksums must match the locally prepared release.
