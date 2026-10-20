# Native code structure review

Current arithmetic follow-up: [source maintenance](SOURCE_MAINTENANCE.md) records
the correction of the double-counted address carry, exact integer division,
shared FP8 grid, two extracted dataflow regions and fresh bounded GPU checks.
Numerical/performance receipts linked below describe their original passes;
their historical outputs are not evidence for this later arithmetic correction.

This review covers all 41 production CUDA modules, their shared helpers, the
small-model graph specialization, packed-resource parsing and portable frame
boundaries. It does not turn partially reconstructed kernels into independently
derived mathematical implementations or change their provenance.

## What changed

| Concern | Change | Preserved contract |
| --- | --- | --- |
| Repeated global vector access | 60 definitions replaced by three shared functions in `cached_vector_access.cuh` | Cache-all vs global-cache loads, streaming stores, 16-byte alignment and optional bounds checking |
| Repeated synchronization helpers | 35 definitions consolidated into seven functions in `memory_checks.cuh`, `copy_barriers.cuh`, `warp_collectives.cuh` | Active masks, selected-lane fallback, volatile operations, async proxy fences, expected transaction bytes and arrive/wait order |
| Repeated active-lane broadcast | 13 definitions consolidated into one in `warp_collectives.cuh` | First active lane, not an assumed lane zero |
| Duplicated scalar work | 67 signed-magnitude expressions use one tested bit contract; ten adjacent duplicate magnitudes and nine duplicate votes share their result | Unsigned wrap, including the magnitude of INT_MIN; votes only combined at unchanged active masks |
| QKV scheduling expressed as a register stream | Named signed-division helper and separate quotient, lane-election and barrier setup | Reciprocal rounding and integer corrections; defined zero-divisor result; full graph output checks |
| QKV matrix variables used as pointers | Weight-address low/high words have a separate lifetime, ending before matrix accumulation starts | Original address calculation and load order |
| Small-model raw byte patches | Fixed-width ABI prefix types and `offsetof` assertions in `window_kernel_abi.h`; named stage/block boundaries | Compact/wide fused geometry and split QKV/projection/pooling shifts; graph fingerprints |
| Packed model records | Named field offsets, lengths and limits in `resource_format` | Payload starts at byte 20, little-endian records, device tags 0/1, authenticated tensor catalogue |
| Style literals | Named immutable style profiles | Existing exposure/contrast/saturation order and activation rules; no invented learned-feature semantics |
| Frame boundary shorthand | Named reflect-once-then-clamp and staged FP16 centering policies | Far padding is NOT periodic reflection; every intermediate half conversion remains observable |
| Final padding coordinate streams | Shared `BoundaryVectorLayout`, prepared unsigned divisor and clear-operation helpers | Component/row/column order, word units, signed widening, legacy carry, per-thread store order and original store policy |

Identical helpers were compared before consolidation. The scalar four-load/store
variant remains separate from the `uint4` variant. Variant-specific async memory,
half permutations and FP8 conversions were not merged by matching signatures.

## Where to read

- `src/native/kernels/production/common/cached_vector_access.cuh`: global vector memory policy.
- `src/native/kernels/production/common/warp_collectives.cuh`: active-mask collectives.
- `src/native/kernels/production/common/copy_barriers.cuh`: async-copy synchronization.
- `src/native/kernels/production/common/word_arithmetic.h`: CPU/device bit contracts and address units.
- `src/native/kernels/production/common/signed_division.cuh`: named corrected-reciprocal tile division.
- `src/native/kernels/production/common/token_dispatch.cuh`: flattened work assignment and shared-tile copy setup.
- `src/native/kernels/production/common/boundary_clear.cuh`: final column-padding work and store-policy adapter.
- [Boundary kernel maintenance](BOUNDARY_KERNELS.md): coordinate derivation, worked example, preconditions and extension checklist.
- `src/native/kernels/production/optimized/global_attention/transformer/query_key_value.cu`: separated setup/data lifetimes.
- `src/native/model/window_kernel_abi.h`: architecture-independent packet prefixes, not host-pointer structs.
- `native_runtime/frame_contract.h`: boundary and precision policies shared with tests.

## Scheduling follow-up

Seven kernels now share `TokenTileWork` (block group and tile index) and
`CopyWarpSchedule` (leader-warp decision, barrier offset and encoded wait words):
global QKV, projection, waiting projection, FFN expansion, publishing expansion,
FFN contraction, and decoder input upsampling. These are inline scalar contracts,
not new dispatches, allocations or a different transport.

Six more reciprocal/correction schedules were replaced by the shared signed
division helper. Seven sign/bias/shift sequences use an explicitly truncating
power-of-two division instead of spelling out the compiler's expansion. The
tile remainder is now a named work coordinate rather than a delayed subtraction
through a reused matrix word.

Copy barriers are derived from `shared_bytes - barrier_count * sizeof(uint64_t)`;
alignment and capacity are checked at compile time. The source guard also checks
the catalogue's `blockDim.z=1` assumption, rather than silently treating a 3-D
block as a 2-D block. Model-free GPU tests cover four layouts, partial warps and
warps with no block leader. The existing wait-word encoding and synchronization
primitives are unchanged.

The publishing FFN also retains two selector words that remain live after setup.
They must not be mistaken for dead division temporaries simply because their old
names start with `packed_half`. Their later uses were traced before extraction.

The original validation receipt below remains separate from this follow-up's
numerical/performance receipt; neither is relabeled as a different source build.
The [follow-up results](benchmarks/current/kernel-scheduling.md) compare against
the first structure pass: all tested reset/history hashes match, and 1080p/4K
medians changed by -0.269% to -0.017%. These small changes are normal timing
variation, not claimed speedups. The checks include 18,600 GPU scheduling cases,
1,000,132 division cases and 1,200,041 word/address checks on each of x86/x64.
Compiled shared-memory, local-memory and stack use are unchanged. Six modified
modules retain their register counts; QKV uses 168 rather than 167 registers per
thread, without changing the driver-reported maximum active blocks/SM. The
resource receipt distinguishes that theoretical capacity from achieved occupancy.

## Boundary iteration follow-up

The input-preparation epilogue and four window-downsample epilogues now share
`common/boundary_schedule.cuh`. Fifteen expanded reciprocal/division regions
(825 lines before extraction) become named boundary iteration calculations.
The shared unsigned quotient implementation lives in `unsigned_division.cuh`;
it now shares exact integer arithmetic with the signed tile helper.

The caller already owns one valid item. Given the next strided index, the helper
counts further items before the exclusive extent. The caller retains its
existing current-item increment, 0..3-item tail and four-item unrolled groups.
This isolates the loop-count contract without changing the swizzled addresses,
optional-resource guards, memory writes, barriers or tensor arithmetic.

Fixed-size tile bounds retain unsigned ordering; flattened extents retain
signed-word ordering. The calculation subtracts one before dividing instead of
adding `stride - 1`, preserving overflow behavior. Integer multiplier preparation,
one remainder correction and zero-stride fallback bits are explicit and covered
by the model-free GPU tests. Zero stride is an
arithmetic compatibility test, not a supported launch geometry.

The [boundary validation report](benchmarks/current/kernel-boundary.md) retains
separate before/after output hashes, resource counts and timings against the
scheduling pass. Its GPU contracts include 1,000,144 full-range unsigned
divisions and 2,017,474 boundary-count checks, in addition to the earlier signed
division and scheduling checks. Historical reports remain unchanged.

## Coordinate and padding-clear follow-up

The final column-padding regions in five kernels now use `BoundaryVectorLayout`
and `clear_boundary_rows`. These regions previously occupied 1,894 lines of
interleaved coordinate, quotient and address words. Their call sites now name
the padding width, first item, exclusive extent and block thread count, and label
each layout field's unit. Earlier tile/edge publication and synchronization stay
in the original functions.

`prepare_unsigned_word_divisor` creates the width's exact integer multiplier
once, outside both loops. The shared clear retains the 0..3-item prefix and
four-item groups. Input preparation supplies its existing checked PTX store;
the window kernels supply their existing volatile pipeline store. The compile-time
callable adds no virtual dispatch, host callback or new allocation. The store
primitives remain distinct because their checked-access behavior differs.

`boundary_layout.h` makes component/row/column roles and word-versus-byte units
explicit. It retains separate signed-product widening and uint32 origin wrap;
full-width scaling now adds each carry once. The [maintenance guide](BOUNDARY_KERNELS.md) explains
these rules with formulas and a worked address example, plus how to extend a
specialization without copying a register stream.

The model-free GPU recorder compared 55,850 clear fixtures and 1,466,440 emitted
word addresses, including their per-thread order, against direct CPU enumeration.
It does not dereference synthetic addresses. Both x86 and x64 passed 1,300,050
word/layout checks. The [separate coordinate report](benchmarks/current/kernel-coordinates.md)
records actual model-output hashes, compiled resources and paired GPU timings
against the boundary-iteration pass; historical reports remain unchanged.

## Deliberately retained low-level code

Low-level operations are not all accidental complexity. Matrix instructions,
FP8/FP16 conversions, half-pair atomics, volatile device flags, cache policy and
async barriers encode numerical or concurrency behavior. They remain explicit
behind operation-specific helpers where their roles are established.

The earlier carry-preserving implementation returned `0x200000000` for
`scaled_word_address(0, 0x40000000)`. The current source corrects this to
`0x100000000`, consistent with its word-offset units. This local correction
does not assert that any external implementation has the same defect.

The attributed model layouts and their 16-byte gaps are unchanged. No upstream
license/provenance comments or historical numerical evidence were removed.

## Remaining semantic reconstruction

This is a substantive structure pass, **not a claim that all instruction-oriented
code has been eliminated**. Fused compute bodies still contain indexed words,
matrix fragments with multiple uses, expanded division schedules and manually
scheduled tensor operations. A global rename into plausible tensor names would
hide uncertainty rather than resolve it.

`tools/audit_kernel_structure.py` inventories every production source/header and
locates these patterns. Its counts are review locations, not a bug count:
matrix fragment indices can be appropriate for a fixed instruction tile. The
shared division helper is integrated into seven scheduling paths; remaining
expanded schedules are not silently declared equivalent.

The current inventory still locates 23 inline reciprocal-bias sites: 15 in
earlier input/downsample edge-coordinate work, six in split-head kernels, and
two in later global attention/QKV work. The boundary loop counts and final
column-padding clears have been extracted. The earlier coordinate divisions
and their live temporaries still need their own derivation, not a blind `/`
substitution or a claim that the whole kernel is now high-level code.

Further work should follow one data-flow region at a time:

1. Establish live-in/live-out values, tensor coordinates and synchronization edges.
2. Separate address/index/fragment lifetimes instead of renaming a physical word globally.
3. Introduce a small operation/geometry contract only where it removes real duplication.
4. Test edge values, packed rounding, padded/odd extents and reset/history output.
5. Compare machine code and paired GPU timings before accepting changed scheduling.

No whole-network algebraic rewrite is justified by a successful compile alone.

## Reproduce the checks

```powershell
python tools/audit_kernel_structure.py --check --output artifacts/current/kernel-structure/source-audit.json
python tools/test_kernel_structure.py
# Explicit, model-free GPU integer probe:
python tools/test_kernel_structure.py --gpu
# Explicit GPU inference; supply two separately built model-free bundles:
python tools/benchmark_kernel_structure.py --before <baseline-bundle> --after <candidate-bundle>
```

The contract runner tests x86/x64 and removes temporary executables. The paired
runner compares fixed-seed reset/history output hashes and alternating-order GPU
timings for Original and Lightweight FP8 at 320x180, 801x603, 1920x1080 and
3840x2160. It reuses the external authenticated model without copying it into the
builds. Exact tested output is a bounded parity result, not proof for every input,
GPU or driver. Games and installed configurations are untouched.

Current receipts are local under `artifacts/current/kernel-structure`. A compact,
path-free [numerical/performance summary](benchmarks/current/kernel-structure.md)
records the measured before/after results. All tested outputs matched; 1080p/4K
GPU medians changed by -0.544% to +0.144%, with no meaningful regression observed.
These small differences are not claimed improvements.
