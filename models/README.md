# External Model Storage

Documentation only: no NVIDIA DLSS5 DLLs or weights are shipped. This refactor does
not relocate or overwrite existing external DLLs.

Use versioned folders in an external store, keeping the original vendor filename:

```text
Models/
  nvidia-dlss5/
    310.8.0.0/
      nvngx_dlssnr.dll
    310.9.16.0/
      nvngx_dlssnr.dll
```

Versions coexist. Authenticate exact contents with SHA-256; folder names/version
strings are not trusted. Never replace a version with a different file. Configure
an absolute path, checksum and separate model ID (`original` or
`small-fp8`). One DLL can contain several models: do not duplicate the entire
container per model or assume a newer DLL automatically selects its newer model.

Extracted compatible resources retain their `model.json` contract. Keep model
identity/architecture separate from container provenance. Local path-bearing
manifests and configurations belong outside source/public packages.

See [architecture and loading](../docs/MULTI_MODEL.md) for implementation status.

[Current model/backend support](../docs/SUPPORT.md) and [two-model measurements](../docs/benchmarks/current/README.md) keep source availability, device validation and package status separate.
