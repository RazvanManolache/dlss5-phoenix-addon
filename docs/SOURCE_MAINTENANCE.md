# Source maintenance and numerical contracts

This work fixes source-maintenance gaps and independently checks numerical
contracts. Existing upstream notices and model-format attribution still apply.
Generation, testing and refactoring do not establish a different origin for
retained code or model knowledge.

## Current changes

| Area | Current behavior and verification |
| --- | --- |
| Address scaling | `scaled_word_address` scales words to bytes once at 64-bit width. Independent byte-domain references check large offsets, negative strides, low-word carry and address wrap. |
| Integer geometry | CPU/device integer multiplier preparation and one remainder correction replace the floating reciprocal construction in the shared division helpers. Signed division supports every int32 bit pattern, including INT_MIN divisors. |
| FP8 publication | Generic and optimized paths share `src/backends/hip_sycl/fp8_grid.h`. The generic path now has ties-to-even behavior independent of the ambient CPU rounding mode. |
| Residual rounding | `upsample_residual` takes a named rounding policy. The existing 512-channel boundary remains explicit and is tested before final FP8 publication. |
| Native dataflow | Attention tail correction is a named scalar operation with a derived exact coefficient. Input preparation separates its optional input handle from its floating scalar lifetime. |
| Layout generation | Both model headers regenerate from maintained declarations, without weights or external tools. |
| Source integrity | Current maintained bytes and separately supplied import bytes have distinct verification operations. Neither operation authenticates historical authorship. |

The earlier pass also removed unused incoming-function resolver state, folded
one finite constant FMA and corrected portable history validity. History
reprojection now checks coordinates whether the optional mask is absent or
all-valid. Conditioning and resolve histories remain independent.

## Address and integer geometry

Offsets in `word_arithmetic.h` count uint32 words. The contract is:

```text
byte_address = base_address + word_offset * 4    (modulo 2^64)
```

The old helper's 64-bit shift already contained the low-word carry; adding that
carry again skipped 4 GiB for some large or negative offsets. For example:

| Base | Word offset | Previous result | Corrected result |
| --- | --- | --- | --- |
| 0 | 0x40000000 | 0x200000000 | 0x100000000 |
| 0x100000000 | -1, represented modulo 2^64 | 0x1fffffffc | 0xfffffffc |

This deliberately changes those address cases. It does not claim that every
synthetic offset names a valid allocation. Tests record synthetic addresses
without dereferencing them. Boundary-layout references scale each coordinate
term directly in bytes, rather than calling the production helper under test.

`integer_geometry.h` prepares `m=floor((2^32-1)/d)` for nonzero unsigned `d`.
For `n<2^32`, multiply-high gives a quotient at most one below `floor(n/d)`;
one remainder correction makes it exact. No floating reciprocal, float-to-int
conversion or floating rounding mode participates. Prepared widths remain
outside coordinate loops. Signed division uses unsigned magnitudes and restores
the quotient sign, preserving defined bits for INT_MIN/-1. A zero divisor keeps
the explicit UINT32_MAX sentinel; production clear operations still require
positive widths and nonzero block strides.

## FP8 and residual policies

FP8 returns a finite E4M3-grid value in FP32 storage, saturating at +/-448.
The preserved special-value policy is NaN -> +448, exact -0 -> +0, and negative
tiny nonzero values -> -0 when rounded to zero. These are tested contracts.
The independent oracle enumerates finite E4M3 values, compares distances and
selects the even code at ties; it does not reuse production masks or helpers.
Tests include every binary16 input, both sides of rounding ties, random FP32
bits, signed NaNs/infinities/zeros and all four CPU rounding modes.
The rounding-mode regression was also run against the pre-change generic
operator header: under FE_UPWARD, 1.0625 incorrectly became 1.125 instead of the
ties-to-even value 1.0. That baseline failed the regression as expected.

The 512-channel residual schedule rounds the skip product to binary16 before
addition; other widths fuse the multiply-add before the final half rounding.
For `projected=-(1+2^-9)` and `skip=gain=1+2^-10`, those paths produce 0 and
2^-20 respectively. Both may subsequently publish FP8 zero, so testing only
that final output would miss a regression. CPU and CUDA tests invoke actual
operator descriptors and check the intermediate and optional output separately.

The Metal generator embeds the shared FP8 helper and depends on it in CMake.
Generated Metal arithmetic is checked against the independent oracle on the CPU.
That simulation is not an Apple compiler or device validation.

## Bounded native dataflow extraction

`common/attention_tail.cuh` subtracts a scaled tail count from a half pair,
retaining FP32 round-to-nearest/FTZ multiplication and half conversion before
half addition. The previous fixed constant path clamped half bits 0x3ed6 to
[0x3dc2,0x3fe9], leaving them unchanged. Its bit transform produced 0x2d60,
which decodes exactly to 43/512. The helper expresses that coefficient directly.
A CUDA differential test compares the previous expression with the helper over
165,536 sum/count pairs, including full-width count boundaries and nonfinite
half lanes. Input preparation also uses a typed float output instead of aliasing
an integer word as a float reference in the extracted scalar calculation.

These are bounded extractions. The large matrix/reduction bodies across the 41
modules still contain instruction-oriented dataflow and some expanded reciprocal
sequences. This work does not claim a complete algorithm-level rewrite, removal
of all register reuse, or a new derivation of the entire native implementation.

## Reproducible model layouts

The maintained declarations are `tools/model_layouts/original.json` and
`tools/model_layouts/small_fp8.json`. The generator validates attribution,
encodings, extents, ordering, unique fields and packed shapes before writing.

```powershell
python tools/generate_weight_layouts.py --check
python tools/generate_weight_layouts.py
python tools/generate_weight_layouts.py --model small_fp8 --check
```

Both current models share 23 packing families and 157 parts, including the
explicit packing gaps. Graph topology and resource selection remain separate.
Small-FP8 regeneration was checked to preserve all executable header tokens.
The declarations preserve existing model knowledge and the attribution in
`third_party/portable_nr/PROVENANCE.md`; generation does not claim independent
discovery or prove every possible external model resource is compatible.

## Maintained bytes and declared import bytes

`third_party/compatibility_bridge/import_manifest.json` distinguishes:

| Field | Meaning |
| --- | --- |
| `source` | Preserved declared import path |
| `import_sha256` | Preserved declared import-input digest |
| `path` | Maintained repository-relative destination |
| `sha256` | Current destination bytes |
| `normalized_lf_sha256` | Current content after CRLF normalization |

```powershell
python tools/verify_source_provenance.py
# After reviewing an intentional maintained-source change:
python tools/verify_source_provenance.py --refresh
# Read-only check of a separately supplied original source tree:
python tools/verify_source_provenance.py --import-root PATH_TO_SOURCE_TREE
```

Compatibility builds reject changed maintained bytes before packaging.
Refresh preserves import hashes, paths, snapshot declarations and notices.
`--import-root` checks exact original bytes separately; it rejects path escapes,
missing files and digest mismatches. It cannot be combined with refresh and
never changes the supplied tree. A matching local snapshot would establish
agreement with the declared digest, not historical authorship or authenticity.

The original import bytes remain unverified: the declared host digest matched
none of 18 local host-source candidates examined during this follow-up. The
nearby legacy Feeder tree also failed the declared host digest. No replacement
import hash or provenance claim was invented to make the check pass.

## Validation and repeat commands

CTest includes integer geometry, word/address contracts, numerical contracts,
model layout generation, compatibility inputs, maintenance tooling and existing
CPU operator/frame checks. Build the focused targets with the normal CMake
configuration, then select tests explicitly. For example:

```powershell
cmake --build BUILD --target integer_geometry_tests kernel_word_contract_tests portable_numeric_contract_tests portable_platform_tests
ctest --test-dir BUILD -R 'integer_geometry_contracts|kernel_word_contracts|portable_numeric_contracts|portable_platform|layout_generation|source_maintenance_tools' --output-on-failure
# Model-free x86/x64 contracts; three CUDA probes run only with --gpu:
python tools/test_numeric_contracts.py --gpu
```

The runner verifies its temporary build path and removes that directory.
GPU execution is opt-in and must
follow the repository's explicit-request rule.

The current compact receipt is `artifacts/current/source-maintenance/validation.json`.
Current checks passed on x86/x64 and the local CUDA device. All 41 production
modules rebuilt. A freshly compiled native API and that rebuilt bundle ran both
models at 320x180 and 321x181: reset, history continuation, then reset again,
12 full-network frames in total. All outputs were finite and the reset outputs
were bitwise repeatable. The receipt binds checks to source, binary and graph
hashes and records temporary cleanup status.

These fixtures do not establish retail-game behavior, visual quality, vendor
parity, large-resolution coverage, performance improvement, or HIP/SYCL/Metal
device equivalence. No game, installation, deployment or publication occurred.
