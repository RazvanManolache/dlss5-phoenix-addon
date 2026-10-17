# Experimental Native x86 Vulkan Runtime

The existing x64 helper remains the default and fallback. This is an additional
Phoenix execution route, not a replacement for that helper or an attempt to load
NVIDIA's x64 DLL inside an x86 game.

## Scope

The portable runtime can now build as x86 and run its Vulkan neural graph inside
that same process. This checkpoint validates the standalone library. It does
**not** yet install an alternate game add-on, change automatic routing, or prove
capture/presentation inside a game. The installer and helper package are unchanged.

The implementation uses the normal 32-bit system Vulkan loader and driver.
On the tested RTX 5080, that driver supports the required portable compute path
but does not advertise the CUDA-kernel launch extension to x86. The portable
SPIR-V operators are used; no hidden CUDA execution or NVIDIA model-DLL code is
substituted for them. Model weights are read from the user's configured external
source as with the existing portable runtime.

## Implementation

- `OperandAddress` retains full 64-bit logical tensor addresses on x86. Checked
  conversion rejects a logical GPU address that cannot be a real host pointer.
- The established x64/ARM64 pointer representation is unchanged. The resident
  operation remains 96 bytes, with its family at byte 40 and scale at byte 88.
- Graph scratch planning, subviews, arena relocation, Vulkan buffer IDs and
  fused-operation packets no longer narrow these logical addresses to x86
  `uintptr_t`. Graph tensors exceeding the host-addressable extent fail explicitly.
- Explicit export definitions preserve the public C ABI names on Windows x86.
- The validation builder derives x86 Vulkan stdcall import aliases from the
  compiler's unresolved symbols. Calling conventions and argument byte counts
  are not guessed or hand-maintained. The real system loader still supplies the
  functions; no proxy driver, registry change or system-DLL patch is installed.

## Reproduce

Use MSVC x86/x64 tools, the existing authenticated weight-source configuration,
[Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) 1.4.365 or newer,
and [glslang](https://github.com/KhronosGroup/glslang) 16.6.0 or newer with its
optimizer. The dependency directory contains `headers/include` and
`glslang/bin/glslang.exe`.

From the Phoenix source root:

```powershell
# CPU-only address, layout, overflow and scratch contracts, both architectures.
python tools/validate_x86_vulkan.py --stage contracts
# Explicit GPU operator tests, both architectures; no full-model inference.
python tools/validate_x86_vulkan.py --stage operators --run-gpu --deps PATH_TO_DEPS
# Full operator/API/temporal-frame comparison, including 1080p.
python tools/validate_x86_vulkan.py --stage frames --run-gpu --deps PATH_TO_DEPS `
  --weight-source PATH_TO_WEIGHT_SOURCE_JSON
python -m unittest discover -s tools -p test_x86_vulkan.py -v
```

The `frames` stage runs three temporal frames at each of `64x64`, `129x97` and
`1920x1080`. Inputs are synthetic, with depth, fractional-pixel motion, validity
masking, reset and subsequent history reuse. Every completed frame must report
the expected process bitness, Vulkan execution, nonempty graph and changed,
finite output. Prepared inputs, raw GPU neural heads and next-frame histories
must match byte-for-byte. The final host-resolved display is checked separately:
either byte-exact, or at most two float32 ULPs with differing host history-gate
hashes proving where the difference arose. Any neural/preparation/history
divergence, larger output error, or unexplained display divergence fails the gate.
The receipt explicitly distinguishes these outcomes; host rounding is not called
byte-exact output. This is comparison with the portable x64 graph, not a claim of
native NVIDIA DLL parity or artistic quality.

An opt-in `validation_stage_hashes` runtime setting exposes these stage digests
in status. It is off by default and has no per-buffer hashing cost during normal
execution. It does not record or write the source images.

The existing operator contracts separately cover matrix tails/strides, packed
weights, publication rounding, fused attention/residuals, normalization and
feature preparation. Public C compile/link/error contracts are also run for
both architectures.

### Validated Checkpoint

On the tested RTX 5080, the standalone x86 and x64 runs each completed all nine
frames: three per resolution listed above. C API, address/scratch and operator
contracts passed on both architectures. The additional Python tooling and
WOW64/probe-interpretation regressions passed 22 tests.

Prepared input stages, all nine raw neural heads and all next-frame histories
matched exactly. Of 18,811,881 final RGB float samples, 39 differed: maximum
absolute error `5.96046447753906e-8`, maximum ULP distance 2. Stage digests isolated
this to the CPU history-gate/resolve boundary; the GPU graph and committed history
did not diverge. Final display output is therefore **not byte-exact** across the
two host architectures. No arithmetic policy was changed to conceal that fact.

The candidate is still a standalone library validation, not game presentation
evidence, and no comparative frame-rate claim follows from these tests.

## Current Outputs

- `artifacts/current/portable-x86-validation.json`: compact build/test receipt,
  source hashes, frame evidence and numerical comparison. Generated, not committed.
- `artifacts/current/portable-x86/`: one model-free alternate x86 runtime, retained
  only after successful full-frame comparison. No helper is overwritten.

The candidate contains the portable runtime, Vulkan backend/provider, CPU
reference backend, `phoenix_frame.exe`, SPIR-V files and license notices. It does
not contain model/NVIDIA DLLs, captures, logs or private configuration. Temporary
builds, x64 controls and raw comparison outputs are removed on completion/failure.
An older successful candidate is not replaced by a failing validation run.

Use `phoenix_frame.exe --vulkan-devices ABSOLUTE_RUNTIME_FOLDER` to enumerate
devices, then follow [the standalone library job format](LIBRARY.md) for an
explicit Vulkan job with your external weight source. The candidate requires
the matching Microsoft C/C++ runtime. Do not mix x86 and x64 backend DLLs.

## Add-on Integration

An opt-in portable route is now wired into the compatibility add-on. See
[Execution Mechanisms](EXECUTION_MECHANISMS.md) for panel options, runtime layout,
fallback policy and the CPU-only integration verification command. Automatic
routing and the current production package remain unchanged. This integration
still requires graphics-fixture presentation validation before promotion.

## Before Automatic Selection

1. Keep the explicit experimental local-Vulkan selection. Preserve automatic helper
   routing and the ability to force the helper. NVIDIA-DLL processing still needs
   its existing supported route. Do not set `x86_native_validated` from this
   Vulkan result: that existing guard protects the different native NVAPI path.
   Portable Vulkan needs its own capability and routing decision.
2. Reuse the established fenced readback/upload lifecycle and output composition;
   run library loading, graph creation and inference away from the render thread.
   Do not label a host-plane roundtrip as zero-copy GPU interop.
3. Preserve live controls, comparison modes, original bypass and history/reset
   semantics. Record actual processing location independently from frame counters.
4. Validate presentation, resize, context/device loss, cancellation, shutdown,
   allocation failure and safe fallback in the graphics fixtures before games.
5. Check actual x86 game address-space headroom. The standalone validation client
   is large-address-aware; not every existing game is. Smaller frames passing
   does not establish that every large game/VR layout fits.

The portable Vulkan route has a different performance profile from the optimized
native-kernel helper. Removing the helper does not by itself establish lower
latency or a higher processed-frame rate. No installer default or performance
claim changes at this checkpoint.

See [the WOW64 investigation](WOW64_EXPERIMENT.md) for the separate cross-mode
experiment and its console/TLS initialization failures.
