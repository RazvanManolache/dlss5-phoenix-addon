# 0.81 Release Validation

## Fresh Checks

| Check | Result | Boundary |
| --- | --- | --- |
| Full x86 and x64 ReShade add-on compilation | Passed | Retained embedded helper/runtime payloads |
| Portable bridge, backbuffer routing, policy and IPC contracts | 531 checks passed per architecture | CPU/fake runtime; not game presentation |
| Native x86, benchmark, UI layout and release guard Python regressions | 76 tests passed | No model or GPU execution |
| Preserved Vulkan shader/import regressions | 6 tests passed | Source/generator tests, not Android device testing |
| Standalone Windows x64 CMake build | Passed | CPU/reference runtime and test components |
| CTest: address contracts, FP8 predecode, C API, platform, CPU operators | 5 suites passed | Generated Metal arithmetic exercised on the host, not an Apple GPU |
| Source publication audit | Passed | Git-visible maintained files; no external model assets |
| Embedded runtime inspection | Passed | Decompressed payloads and nested add-on resources checked |

The CMake check exposed a Metal-generator regression from the widened logical
operand address type. Metal now explicitly reconstructs its five device-pointer
fields; the host arithmetic and generator regression tests pass after the fix.

## Retained Evidence

The [execution benchmark export](benchmarks/0.81/README.md) is based on the
validated pre-release measurements, with receipt hashes in the public JSON.
It is not a fresh timing run of the release binaries. Optional standalone
runtimes are retained hash-verified builds rather than newly rebuilt SDKs.

Earlier packaged-client tests cover real Phoenix/NVIDIA inference, live
strengths, refinement, switching, comparison and worker retirement with the
local native x86 bootstrap profile. Their raw receipts stay local. They do not
establish arbitrary game presentation, resize/device-loss behavior or support
on other Windows/driver profiles.

## Publication Scope

No retail game was launched for this release preparation. User configurations,
game profiles, logs, captures, credentials, model weights and NVIDIA model DLLs
are excluded from the archives. Source archives come from the audited Git tree,
not a recursive copy of the working directory. Public authorship uses a GitHub
noreply identity.

Four retained embedded binaries contain their former **project-source checkout
root** in assertion source-location strings. This narrow diagnostic exception
does not permit usernames, installed-game paths, configs or raw JSON reports.
The release audit tests enforce that distinction. No verified executable is
binary-patched merely to relabel its build location.

The optional WOW64 download includes unmodified dynamically linked WOW64Ext,
the LGPL-3.0 license text, its pinned corresponding source and rebuild recipe.
The Windows/driver bootstrap hash/profile is functional compatibility metadata,
not a personal game profile. A mismatch must reject the experimental path.

Archive inventories, SHA-256 checksums and the publication commit are in the
downloadable `FILES_SHA256.json`, `SHA256SUMS.txt` and `VALIDATION.json`.
