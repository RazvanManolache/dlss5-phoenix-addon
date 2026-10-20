# Register pressure and spill experiments

The historical optimization in `f86c2f2` tuned the Original model using the 310.8
DLL. It selected some register budgets and two resolution-dependent entries.
The current two-model recheck is documented in
[the measured report](benchmarks/current/register-limits.md); the older
[investigation](NATIVE_MEMORY_INVESTIGATION_2026-10-13.md) retains its own inputs,
results and conclusions rather than being relabeled as a new measurement.

## What the terms mean

- **Register budget:** a compiler allocation limit, not a promise that every
  thread uses exactly that many registers. The report separates requested limits
  from the register count of the assembled function.
- **Spilling:** live values stored in thread-local memory when the allocator
  cannot keep everything in registers. It can affect latency but is not, by
  itself, register corruption or an out-of-bounds write.
- **Stack/local bytes:** compiler/driver resource quantities for a function and
  thread. They are not host RAM or the whole model's GPU-memory allocation.
- **Spill load/store bytes:** static compiler-reported quantities, not a measured
  number of bytes transferred per frame. Do not multiply these counts by the
  image area to infer bandwidth or total VRAM consumption.
- **Maximum blocks per SM:** a theoretical CUDA Driver occupancy calculation
  using the actual block size and zero dynamic shared memory. It is not a
  measurement of achieved occupancy, cache behavior or execution overlap.

Do not interpret `cuobjdump`'s `LOCAL:0` field alone as "no spilling." In these
binaries it can coexist with a nonzero `STACK` and nonzero PTXAS spill counts.
The inventory therefore retains the compiler's stack/spill fields and the
driver's local-storage attribute separately, with the tool/source of each field
explicit. Resource fields from different tools need not have identical accounting.

Fewer spills are not automatically faster: more registers can reduce residency,
and the allocator also changes instruction scheduling. Higher theoretical
occupancy is not automatically faster either. Compare complete network GPU
intervals, then investigate individual kernels if a repeatable difference warrants it.

## Production and experiment boundaries

`tools/production_build_native_kernels.py` has no experimental tuning flags.
Its catalogue, architecture, assembler policies, block sizes and source budget
annotations define the production bundle. Do not change them just to run a test.

`tools/benchmark_register_limits.py` takes a freshly built bundle and creates
temporary variations outside the production catalogue:

1. Inventory all 41 modules, including both entries in the two dual-entry modules.
2. Parse PTXAS properties per entry and query matching CUDA Driver attributes.
3. Prepare PTX for eight explicitly listed spill-bearing kernel families.
4. Assemble unchanged-budget controls and require identical instruction bodies.
5. Change only the chosen `.maxnreg` directives for relaxed/tighter mixes.
6. Separately force default/large entries by editing temporary manifest metadata,
   without recompiling those two kernel bodies.
7. Run both models with the full default schedule and matched inputs; compare
   output hashes before considering a profile's timings valid.
8. Remove the temporary variants and retain compact hashes/resources/results.

The fixed eight-target list makes the investigation bounded and reproducible.
It is not an automatic optimizer or a claim that all spilling kernels were tuned.
A multi-kernel mix cannot identify which individual kernel caused a difference.

The offline probe optionally reports the production selector's chosen entry and
the generated plan's launch count per module. All 41 function handles are created,
but a model can leave some modules unused. A zero launch count must not be
interpreted as that model executing the module. These are generated-plan counts
for a full pass, not a hardware trace of dynamically retired instructions.

The current selector uses actual neural input area, computed in 64 bits, at
network creation. Its threshold is 18,662,400 pixels, not the display size or a
list of supported resolutions. A threshold measured for one model/GPU must not
be assumed optimal for another model/GPU without a separate comparison.

## Correctness and timing

For each exact model/extent, every successful profile must match production's
reset A, one-frame history and reset B output hashes. The probe also rejects
non-finite output, reset-repeatability failures and D3D12 errors. The reporter
independently checks the recorded hashes; failed modes have no public timing or
image result. This remains a bounded synthetic fixture, not moving-scene certification.

801x603 enables D3D12 debug validation. 1080p, 4K and 8K use two fresh processes
per profile in forward/reverse profile order. The sizes just below/at the switch
use one process per profile and are policy/correctness diagnostics, not precision
performance rankings. Both models use the authenticated 310.9.16 DLL without
copying it into experimental folders.

GPU timestamps enclose the full engine workload, barriers and matching output
copy. They exclude uploads, readbacks and CPU gaps. Resource inventory runs
outside these measurements and launches no kernels. There are no per-kernel
GPU timestamps or hardware stall/bandwidth counters in this experiment.

Small differences under unlocked stock clocks need additional isolated runs.
Neither a favorable single mix nor an unchanged output hash justifies a blanket
production budget change. A future selection change needs per-model tests,
threshold/odd-size checks and independent counterbalanced confirmation.

```powershell
python tools/production_build_native_kernels.py register_probe --compiler "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe" --jobs 4
python tools/benchmark_register_limits.py --baseline build/register_probe
python tools/report_register_limits.py
python -m unittest discover -s tests/python -p test_register_limits.py
```

The driver queries and experiments require the NVIDIA CUDA/NVAPI native route.
These are not portable Vulkan, CPU, AMD, x86-transport or installed-game tests.
No Windows registry configuration, game installation or release is changed.
