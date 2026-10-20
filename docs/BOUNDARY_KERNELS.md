# Boundary kernel maintenance

This documents the final column-padding clear in input preparation and the four
window-downsample kernels. It is a tensor-storage operation, not an inference
pass, image enhancement setting or learned feature. Earlier edge/tile publication,
matrix work and synchronization remain in their original kernels.

## Source map

| File in `src/native/kernels/production/common` | Responsibility |
| --- | --- |
| `boundary_layout.h` | CPU/device layout fields and coordinate-to-byte-address contract |
| `integer_geometry.h` / `unsigned_division.cuh` | Prepare an exact integer multiplier once; compute unsigned quotients |
| `boundary_schedule.cuh` | Count further strided items without overflowing an added rounding bias |
| `boundary_clear.cuh` | Split flat indices, group work and invoke the caller's store policy |
| `word_arithmetic.h` | Low/high words and full-width word-to-byte address scaling |

The callers bridge existing launch/register words to named local fields. No
launch packet, graph, weights, allocation ownership or exported entry point changes.

| Caller specialization | Component bits | Vectors per spatial coordinate |
| --- | ---: | ---: |
| Input preparation | 2 | 4 |
| Downsample, 32 channels | 3 | 8 |
| Downsample, 64 channels | 4 | 16 |
| Downsample, 128 channels | 5 | 32 |
| Downsample, 256 channels | 6 | 64 |

## Coordinate contract

One work item clears four adjacent `uint32` words, totaling 16 bytes. The flat
index interleaves component first, then column, then row. Let `C` be vectors per
spatial coordinate, `W` the positive padding width and `i` the flat work index:

```text
component = i % C
spatial   = i / C
row       = spatial / W
column    = spatial - row * W
absolute_column = uint32(column_origin + column)
```

`C` is a compile-time power of two; division/remainder by `C` become shifts/masks.
The width divisor is prepared outside both loops. Each item then uses integer
multiply-high and one remainder correction. Preparation uses
`floor((2^32-1)/divisor)` with integer arithmetic; no float conversion or hardware
reciprocal is involved. The zero-divisor sentinel is handled before division.

`BoundaryVectorLayout` has these units:

| Field | Unit and interpretation |
| --- | --- |
| `base_address` | 64-bit device byte address |
| `component_stride_words` | Distance between component vectors, signed 32-bit word bits |
| `row_stride_words` | Distance between rows, signed 32-bit word bits |
| `column_origin` | Absolute column origin, added modulo 2^32 before signed widening |

Each stride product is widened separately and accumulated modulo 2^64:

```text
word_offset = signed(component) * signed(component_stride_words)
            + signed(row)       * signed(row_stride_words)
            + signed(absolute_column) * 4
address = scaled_word_address(base_address, word_offset)
stores  = address + {0, 4, 8, 12} bytes
```

For example, `C=8`, `W=3`, `i=34` gives component 2, row 1, local column 1.
With origin 2, component stride 256 words, row stride 64 words and base `0x1000`,
the offset is `2*256 + 1*64 + 3*4 = 588` words and the address is `0x1930`.

The current address contract is `base_address + word_offset * 4`, modulo 2^64.
For example, `scaled_word_address(0, 0x40000000)` is `0x100000000`.
The former extra carry counted the overflow of the low offset twice; it also
misaddressed negative offsets in some base alignments. This is an intentional
correction of that behavior. Independent byte-domain references replace tests
which called the same production address helper. Signed-stride/overflow fixtures
test arithmetic only and must never dereference their synthetic addresses.

## Work and store ordering

The caller retains the positive-width and current-index/extent checks. The stride
is the nonzero block thread count; extents and increments must stay within the
supported signed-positive index geometry. The helper counts further items, processes
`(further_steps + 1) & 3` prefix items, then processes four-item groups. The final
group retains the signed-word extent comparison. This is the previous grouping,
not four extra inference iterations or a changed padding extent.

The group computes four independent addresses before publishing its vectors.
Each thread emits vectors in ascending work-index order and emits words at
offsets 0, 4, 8, 12. This does not promise an inter-thread or inter-block ordering.
There is no new barrier, allocation, host callback or indirect device call.

The store callable is a compile-time policy, inlined into the kernel:

- Input preparation retains `write_boundary_word<CheckBounds>` and its PTX store
  and alignment/range checks.
- Window kernels retain `pipeline_write<CheckBounds>` and its volatile stores
  and existing range/error-count behavior.

These primitives are intentionally not merged: their checked-access contracts
differ. Zero-divisor fallback bits are tested for the arithmetic helper, but
zero width/stride are not valid clear-operation preconditions.

## Tests and acceptance

`kernel_word_contract_tests.cpp` checks the layout on both x86 and x64, including
origin wrap, signed strides and an independent byte-domain reference.
`kernel_boundary_clear_tests.cu` records emitted addresses on the GPU and compares
every word and its order against direct CPU integer-division enumeration. It
tests all five component counts, prefix/group boundaries and irregular strides.
Computed tensor addresses are recorded, not dereferenced; the recorder bounds
its own output and reports overflow.

The earlier paired native-runtime check separately compared both models' actual
RGBA16F reset/history outputs, finite values and reset repeatability at 320x180,
801x603, 1920x1080 and 3840x2160. Small sizes enable D3D12 debug validation.
1080p/4K timing rows use alternating before/after fresh processes. Compiled
register/shared/local/stack counts are retained alongside the instruction hashes.
See the [coordinate-refactor results](benchmarks/current/kernel-coordinates.md).
Those receipts describe the earlier carry policy. The current correction and
targeted GPU checks are recorded in [source maintenance](SOURCE_MAINTENANCE.md).

These are bounded synthetic checks, not a guarantee for every tensor geometry,
GPU, driver or interactive game. The source audit still lists other expanded
coordinate divisions; this extraction does not claim those regions are derived.

```powershell
python tools/audit_kernel_structure.py --check
python tools/test_kernel_structure.py --gpu --output artifacts/current/kernel-coordinates
# Build a baseline before editing and a candidate afterwards, in fresh build/ folders.
python tools/benchmark_kernel_structure.py --before build/coordinate_baseline --after build/coordinate_candidate --output artifacts/current/kernel-coordinates
```

When extending a layout, establish its index order, stride units, bounds, store
policy and synchronization separately. Add a specialization and address-order
fixtures rather than copying another register stream. Preserve precision,
ordering and the documented address units, and validate intentional changes.
