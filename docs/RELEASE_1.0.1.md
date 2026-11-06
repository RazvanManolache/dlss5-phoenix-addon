# Phoenix DLSS5 1.0.1

- Set `PHOENIX_DATA_ROOT` to keep shared helper binaries and private sessions on a writable drive of your choice. See [helper storage](GRAPHICS_COMPATIBILITY.md) for setup and the default location.
- Phoenix can open and hash model files and load kernel binaries from Windows paths beyond `MAX_PATH`, including in games without long-path-aware manifests. Model hashing errors include the filename and Windows error code.
- Helper sessions share verified binaries through hard links. Session cleanup preserves the shared files and retained captures.

## Downloads and installation

Install one Windows package with [full-addon-support ReShade](https://reshade.me/). Keep the matching add-on and helper together. The RTX Compatibility package includes experimental kernels for older RTX cards. The separate Input and Portable Runtimes packages are optional. See [installation](INSTALLATION.md) for the file layout.

## Validation

See [validation scope](VALIDATION_1.0.1.md) and the downloadable `VALIDATION.json` for the checks and binary hashes. The 76-game regression used the earlier 1.0 binaries.
