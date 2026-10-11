# Portable runtime source provenance

- Upstream: [mauri870/dlssnr-pytorch](https://github.com/mauri870/dlssnr-pytorch).
- Recorded project reference revision: `7b767ed4ecc014bc9e086d97927f74c9cb2d39da`.
  This is the revision used for this comparison, not a claim that the original
  import commit of every local file was recorded.
- License: [MIT](LICENSE), copyright (c) 2026 Mauri de Souza Meneguzzo.
- Publication attribution: [NOTICE](NOTICE).

## Current retained dependency

The Python runtime modules were removed from this repository on 2026-10-11
after migration to native C++ execution. LICENSE and NOTICE remain because
`native_runtime/weight_layouts.h` contains declarations derived from the
portable runtime's tensor layouts. The sections below record the earlier
Python source subset and its adaptations; those Python paths are historical.

## Earlier source subset

The 16 Python files other than the two adaptations below originated from
the reference revision at their respective upstream paths. The readability
update applies local formatting and modification notices without changing
their Python syntax trees.
LICENSE matches the upstream text and is retained byte for byte.

| Local file | DLSS5 Phoenix adaptation |
| --- | --- |
| `dlssnr/weights.py` | Removed safetensors imports and file load/save methods; retained the in-memory typed tensor views. |
| `dlssnr/tensor_layout.py` | Derived from upstream `dlssnr/extract.py`, rather than a file of this name. Retained tensor layouts and unpacking helpers for supplied runtime records; omitted DLL/resource discovery, archive handling, file export and command-line extraction. The compatibility exception is renamed `ModelCompatibilityError`. |

The two adapted files carry prominent attribution and change notices.
No executable code changed during this publication transfer. No upstream
extraction module or CLI, extracted model weights, NVIDIA DLLs or compiled
kernels are included in this dependency subset. The tensor layout descriptions
and conversion code remain third-party-derived source, as identified above.

## Readability refresh (2026-10-10)

The current runtime sources were copied from the validated DLSS5 Phoenix
readability pass. Original license text, copyright notices and the complete
upstream NOTICE remain retained. Publication-specific adaptation headers are
preserved. No model weights, NVIDIA binaries or extraction tools were added.
