# Native x86 Evaluation Investigation

Status: ordinary x86 NVAPI model execution still fails on the tested system;
safe capability rejection is tested. Experimental same-process native x64
execution and an x86 Phoenix DLSS5 graph host now work in isolated/package probes,
as described in [Execution Mechanisms](EXECUTION_MECHANISMS.md). That workaround
does not make the original 32-bit driver path supported.
The automatic x64-helper fallback remains enabled. Do not remove its safety gate
based on a successful build, exported function, or an API query alone.

## Observed Evidence

- A full x86 add-on builds and includes the Phoenix DLSS5 evaluator.
- The initial local D3D11-on-12 x86 fixture failed during live evaluation.
- The debugger stopped on a first-chance access violation in
  `ntdll!RtlEnterCriticalSection`, below 32-bit `nvwgf2um` and `nvapi_impl`.
  Its exception-context recovery failed and it warned about stack unwinding;
  this is not a complete source-symbolized crash dump.
- The retained x86 companion binary identified by SHA-256
  `0151d110f4f4fb04da9f294c51d0192a49e5f2d32ac13277b3bdf47954b48a18`
  has a return address at RVA `0x43355` in that trace.
- Disassembly immediately before it passes device, blob, byte count and output
  module handle to the first module-lifecycle function pointer. The module handle
  is at offset `0x1c` of the x86 owner. This matches `Api::create_module` and
  `Network::rebind` in `src/native/runtime/kernel_module.h`:
  **`NvAPI_D3D12_CreateCuModule` is the narrowed failing call**, not the later
  `LaunchCuKernelChain` dispatch.
- The x86 capture plus x64-helper path passes live fixture tests. That validates
  the fallback, not native x86 inference.

## Architecture Hypothesis

The test adapter is a GeForce RTX 5080. NVIDIA's CUDA Windows installation guide
says Ada is the last architecture with driver support for 32-bit CUDA
applications. Its NVAPI system requirements still describe general 32-bit
support; those are different capability claims. The local NVAPI header also
marks the CUDA-module APIs experimental and internal.

This makes absent post-Ada 32-bit CUDA support a plausible explanation for the
module-creation fault. It does **not** establish a documented support guarantee
or limitation for this particular experimental NVAPI entry point. A pointer,
lifetime, device-wrapper, or driver issue still needs to be excluded.

Sources:

- [CUDA Windows installation requirements](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [General NVAPI system requirements](https://docs.nvidia.com/nvapi/sysreq.html)
- [Official NVAPI interface identifiers](https://github.com/NVIDIA/nvapi/blob/main/nvapi_interface.h)

## Isolation Checklist

1. Wait for the active compatibility sweep to finish before any GPU probe.
2. Build a small x86 and x64 probe using the same official NVAPI declarations.
   Record pointer/handle sizes, adapter identity and driver version; retain
   symbols for the probe.
3. Use a fresh native D3D12 device without ReShade, hooks, helper transport,
   presentation, or model allocations. Load exactly one verified kernel blob.
4. Bound the probe with a parent-process timeout and capture the exact call,
   API result or second-chance exception. A debugger stop on a first-chance
   exception must not automatically be reported as an unhandled crash.
5. Compare x86 and x64 `CreateCuModule` on that device with identical kernel
   bytes. Check the optional fatbin/PTX capability query independently; a
   non-null function address is not proof that module creation works. Also try
   the older named-cubin-shader creation API in its own process to determine
   whether it offers a viable D3D12 alternative to the module API. Shader
   creation alone would still not validate full model execution.
6. If standalone module creation passes, reintroduce device wrapping and hooks
   separately. If only the x86 driver path fails, do not keep retrying it inside
   the game or claim a compiler-flag fix.
7. Keep native x86 gated until actual module creation, kernel evaluation,
   completion, and presented output all pass. A future D3D12-compute backend
   without the unsupported CUDA-host dependency would be a separate solution,
   not a completed fix today.

No production package was changed for this investigation checkpoint. The
background-only rerun completed using the safe helper package.

Reproducer: `python tools/probe_nvapi_module.py` compiles both probes and reports
interface sizes without loading NVAPI or using the GPU. Only the explicit
`--run-gpu-probes` flag runs the bounded hardware cases. Both modes clean their
temporary binaries and retain one compact current report outside the package.

## Standalone Hardware Results

Using the same verified 53,288-byte kernel blob and fresh native devices:

| Probe | x86 | x64 |
| --- | --- | --- |
| NVAPI initialization | Success | Success |
| Fatbin/PTX capability | API success, unsupported | API success, supported |
| CUDA module creation | Unhandled access violation `0xC0000005` | Created and destroyed successfully |
| Older named cubin shader | `NVAPI_INVALID_IMAGE` (`-232`), no handle | Created and destroyed successfully |
| Guarded module path | Explicit unsupported error; module call not entered | Created and destroyed successfully |

Both architectures also passed CPU tests for the shared guard. These results
exclude ReShade, full-model allocations, hooks, and transport from the isolated
failure. They support the driver/capability explanation but do not establish
which internal NVIDIA DLSS5 implementation defect causes the access violation.

`kernel_module.h` now queries the x86 capability before module creation and
rejects unavailable/failed/unsupported capability queries. The x64 path is not
changed. This prevents the reproduced unsupported call; it does **not** make
native x86 CUDA inference work. The source guard has not been promoted into the
package used for the completed 135-case compatibility rerun. That package already
keeps automatic x86 native processing disabled and uses the validated helper.

## Direct Driver Bridge

**Current update:** the driver-only x86-to-x64 thunk now creates and destroys
the real native CUDA module successfully after early CUDA initialization with
a verified high-address reservation retry. Whole x64 Phoenix DLSS5 and NVIDIA DLSS5 model
outputs also pass in separate x86 children and match normal x64 exactly. See
[the updated native experiment](WOW64_EXPERIMENT.md#latest-result-native-inference-works-in-the-probe).
The subsequent [full x86-host probe](WOW64_EXPERIMENT.md#route-3-x86-host-inference)
now evaluates the complete graph with x86-owned model/plan/controls and a generic
x64 driver executor. Six outputs match the x64 controls exactly, including live
strength changes. Game/add-on integration remains unfinished. The failure
descriptions below are the earlier checkpoint, before these fixes.

This is the requested native-route investigation, not the separate portable
Vulkan implementation. At the historical checkpoint below, it was not yet a
working neural renderer; the linked current results supersede that checkpoint.

The isolated probe now includes an x86 coordinator and a small x64 **driver-only**
thunk in the same process. No x64 Phoenix DLSS5 runtime is loaded in that child.
It initializes real native TLS, creates a real x64 D3D12 device/queue, loads
NVAPI64 and attempts to create the identical verified native kernel module used
by the normal x64 positive control. Calls requiring x86 GDI state are bridged
back to the coordinator. The driver escape bridge marshals the public Windows
request structure and copies its private payload unchanged; it does not invent
kernel packets, patch the display driver or alter system configuration.

Observed results on the installed RTX 5080 driver:

| Case | Result |
| --- | --- |
| Real x86 CUDA driver initialization | `CUDA_ERROR_NO_DEVICE` (100), confirmed by its own error-log API |
| Unguarded x86 NVAPI module call | Access violation in `RtlEnterCriticalSection`; invalid critical-section pointer `0xF4`, below `nvwgf2um` |
| Same-process x64 driver-only thunk | NVAPI initializes; module creation returns `NVAPI_ERROR` (-1), no module |
| Driver escape marshaling | All 13 observed requests reach the real x86 Windows entry point; no module success |
| Native x64 module control | Identical module creates and destroys successfully |
| Separate CUDA-init diagnostic with escape bridge | GPU copy verified; all 14 observed escapes return NT success; `cuInit` still returns 2 |

Repeating the CUDA-init diagnostic with the existing native registry, identity,
thread-TLS, native-image and AppCompat-view variants did not resolve it. CUDA's
log API repeats the generic initialization error; it does not identify an actual
failed allocation. Do not diagnose this as confirmed VRAM exhaustion.

The driver-only probe is a prerequisite for route 3, **not** evidence that the
x86 Phoenix DLSS5 graph or a complete frame ran through it. Route 2 (loading the whole
x64 Phoenix DLSS5 runtime) is distinct. No inference timing is valid for either failed
route. The existing external helper and installed game files remain unchanged.

Reproduce the driver-only experiment, with explicit GPU initialization enabled:

```powershell
python tools/probe_wow64_tls.py --wow64ext PATH_TO_WOW64EXT `
  --run-gpu-probes --try-native-driver-bridge --large-address-aware `
  --case thread_gpu_native_driver
```

The compact receipt is `artifacts/current/x86-driver-bridge-investigation.json`.
Its `driver_bridge_conclusions` distinguish observed same-process driver calls,
module creation and neural inference. The runner's exit code alone is not a pass
for any of those GPU steps. Both architectures run CPU-only ABI and payload-size
contracts first; temporary binaries are removed afterwards.

`tools/probe_nvapi_module.py --run-gpu-probes --trace-driver` reproduces the x86
fault under a debugger and records real x86/x64 CUDA initialization separately.
The first-chance debugger stop is not itself the proof of an unhandled crash;
the independent normal x86 child also exits with `0xC0000005`.

NVIDIA DLSS5 explicitly excludes RTX 50-series and newer GPUs from 32-bit CUDA driver
support. This explains why native x86 CUDA cannot be assumed available; it does
not prove that every possible cross-mode driver bridge is impossible.

- [NVIDIA DLSS5 32-bit CUDA support plan](https://nvidia.custhelp.com/app/answers/detail/a_id/5615/kw/scanning)
- [CUDA error-log API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__LOGS.html)

### Independent Arithmetic Control

`tools/probe_nvapi_module.py --run-gpu-probes --arithmetic` assembles a tiny,
independently authored SM 120 PTX kernel without a model or FP8 instructions.
The normal x64 legacy NVAPI route dispatches it and verifies all 4,096 output
integers. Direct x86 shader creation returns `NVAPI_INVALID_IMAGE` (-232).
The initial unbootstrapped same-process x64 thunk also failed shader creation;
this was not a Phoenix-kernel-only problem. With the native reservation/CUDA
bootstrap, the same-process driver-only thunk now creates and destroys this
independent shader successfully, with NVAPI status 0 and 15 observed bridged
escapes. This thunk check verifies shader creation, not arithmetic dispatch or
full x86-host Phoenix DLSS5 inference. Its separate compact receipt is
`artifacts/current/x86-driver-arithmetic-investigation.json`.

Reproduce that creation-only check with `tools/probe_wow64_tls.py`, specifying
`--run-gpu-probes --try-native-driver-arithmetic --large-address-aware
--cuda-first --case thread_gpu_native_driver_arithmetic`, plus the documented
`--wow64ext` and `--ngx-core` paths. This does not add normal 32-bit CUDA driver
support to Blackwell.
