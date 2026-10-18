# WOW64 Cross-Bitness Experiment

**0.81 status:** the successful probes below now underpin two opt-in add-on
transports. Packaged-client inference and worker tests passed; real game
presentation remains unvalidated. The separate x64 helper and conservative
automatic routing remain available. See [current integration and limits](EXECUTION_MECHANISMS.md).

The experiments recorded below used disposable processes. They did not modify
an installed game, driver installation, registry setting or on-disk Windows
library. Historical debugging and feasibility results are not blanket proof
of safety on other Windows builds or drivers.

## Latest Result: Native Inference Works In The Probe

The previously generic CUDA error 2 was traced to CPU virtual-address reservation
inside CUDA device enumeration. The real `VirtualAlloc2` call requested
8,657,043,456 bytes (8.0625 GiB), `MEM_RESERVE`, `PAGE_NOACCESS`, no base address
and no extended parameters. It succeeded in normal x64, but returned Windows
error 87 (`ERROR_INVALID_PARAMETER`) in the native x64 thread inside WOW64.
This is an uncommitted CPU address reservation, not an 8-GiB VRAM allocation.

The opt-in research shim retries only that constrained class of failed requests
with an explicit high-address requirement, then bounded exact-address attempts
through the real native Windows allocator. It never truncates the size, commits
the region, changes protection, overrides existing parameters or claims a
failed call succeeded. `VirtualQuery` checks a successful retry before returning
it to the driver. Original successful allocations remain unchanged.

Bootstrapping the real CUDA driver with this retry **before creating D3D12/model
state** now permits optimized inference. Enabling the retry without that early
CUDA initialization did not unblock model creation in the tested sequence.
The exact reason for the original API's WOW64-specific rejection remains an
implementation inference; the failed call and successful allocation are measured.

| Isolated check | Current result |
| --- | --- |
| Native x64 CUDA initialization inside x86 | Success, with a real high-address reservation |
| Driver-only x86-to-x64 thunk | Real native CUDA module created and destroyed, without loading x64 Phoenix DLSS5 in that child |
| x64 Phoenix DLSS5 inside x86 | Three finite, changed model outputs; SHA-256 matches normal x64 output exactly |
| x64 NVIDIA DLSS5 inside x86 | Three evaluations; final finite, changed RGB output matches normal x64 exactly |
| Process ownership | x86 executable PID equals inference PID; no graphics helper |
| Tested input | Synthetic 640 x 384, 1:1; same input/model for normal and cross-mode controls |

Phoenix DLSS5's comparison hashes cover RGBA half-float output; NVIDIA's cover packed
RGB half-float readback, excluding texture padding and unspecified alpha.
Neither check is a game-presentation or performance benchmark.

This proves the standalone feasibility of requested **route 2**, for both real
x64 libraries. It also solved the driver-module prerequisite for **route 3**;
the separate x86-host inference result below now extends that prerequisite to
real graph execution. Neither is a production game-transport integration.
Portable Vulkan remains a separate fallback, not either of those native
mechanisms. Keep the working external helper unchanged.

```powershell
python tools/probe_wow64_tls.py --wow64ext PATH_TO_WOW64EXT `
  --run-gpu-probes --try-native-driver-bridge --try-d3d12-frame `
  --ngx-sdk PATH_TO_NGX_HEADERS --ngx-core PATH_TO_INSTALLED_X64_NGX_CORE `
  --large-address-aware --bridge-create-dc --bridge-display-gdi `
  --native-appcompat-view --retry-native-reservation --cuda-first `
  --case thread_gpu_native_driver --case thread_d3d12_cuda `
  --case thread_d3d12_phoenix --case thread_d3d12_nvidia
```

Current receipt: `artifacts/current/x86-driver-bridge-investigation.json`.
Read `model_parity_conclusions` for verified process ownership and exact-output
comparison, and `driver_bridge_conclusions` for the separate module-only probe.
CPU-only regression tests cover reservation eligibility, ABI layouts and false
success rejection. Native TLS/loader, GDI and AppCompat experiments still use
unsupported/private Windows behavior. This is not enabled in automatic routing,
installed games, the application, or a distributable package.

Additional tracing uses `--trace-cuda-init` and optional hexadecimal
`--cuda-break-rva` checkpoints. It requires `pefile`; offsets are driver-specific
and must be re-inspected after a driver change. The receipt pins the inspected
CUDA image's SHA-256. These traces do not override normal-child results.

## Route 2 Performance

Measured on an RTX 5080, driver 616.64, at 1920 x 1080 and 1:1 resolution.
Each backend/process combination initialized independently, warmed for five
seconds, then measured twenty seconds of serial, fence-completed calls. Both
Phoenix DLSS5 and NVIDIA DLSS5 used real native GPU processing, without Vulkan or a helper.
The current run resets each frame to match route 3's validated path.

| Backend | Execution | Mean GPU time | Mean synchronized call | Completed outputs/s |
| --- | --- | --- | --- | --- |
| Phoenix DLSS5 | Normal x64 control (#1) | 4.007 ms | 4.391 ms | 211.95 |
| Phoenix DLSS5 | x64 library inside x86 (#2) | 4.037 ms | 4.508 ms | 204.78 |
| NVIDIA DLSS5 | Normal x64 control (#1) | 4.027 ms | 4.455 ms | 224.34 |
| NVIDIA DLSS5 | x64 library inside x86 (#2) | 4.004 ms | 4.465 ms | 221.70 |

These are isolated GPU-resident texture calls, **not game FPS or installed-addon
presentation latency**. The native worker is entered once; per-frame x86 caller
handoff and capture/presentation integration are not measured. Initialization,
host image transfers, hashing and image validation are outside the timed loop.
Output throughput includes timing collection and status parsing, so it is not
the reciprocal of GPU time. Phoenix DLSS5 GPU timestamps include the model and output
copy; NVIDIA DLSS5 timestamps surround its evaluation. Compare each backend to its
own x64 control, not as a precisely identical cross-backend timing scope.

All four timed children completed normally, with matching process/inference
ownership, no helper, and finite changed output after measurement. Each run
completed more than 4,000 measured frames. The separate fixed-frame parity probe
above remains the explicit exact-output check. Do not directly compare these
isolated output rates with the ReShade fixture's full capture/helper/presentation
output rate.

The measured synchronized-call difference was +0.117 ms for Phoenix DLSS5 and
+0.011 ms for NVIDIA DLSS5 in this run. This is one serial measurement, not a repeated
statistical determination of WOW64 overhead or a guarantee for arbitrary games.

Reproduce with the native experiment command above, omitting
`--try-native-driver-bridge`, `--case thread_gpu_native_driver` and
`--case thread_d3d12_cuda`, then adding `--benchmark --benchmark-reset`. The retained receipt is
`artifacts/current/wow64-method-benchmark.json`; `benchmark_conclusions` refuses
timings for crashed, wrong-process, incomplete or unvalidated runs. Production
add-ons, automatic routing, the helper and game installations remain unchanged.

## Route 3: X86 Host Inference

The remaining blocker was a missing interface, not another reproduced kernel
crash: the original driver thunk stopped after creating/destroying one module.
It had no resource/binding/dispatch channel for an x86 Phoenix DLSS5 host. Normal x86
NVAPI remains unsupported here, and x86/x64 COM interfaces cannot be exchanged
as if they shared a vtable layout.

The new disposable host compiles Phoenix DLSS5's existing bundle authentication,
generated graph, native plan preflight, relocation and control patchers into
the **x86 coordinator**. It sends fixed-width packets to an x64 native thread in
the same process. GPU addresses and descriptor encodings remain 64-bit opaque
values. The driver executor owns native COM objects, module/function lifetimes,
uploads, raw dispatches, barriers, fences and readback. It contains no Phoenix DLSS5
graph generator, neural controls, model loader or x64 Phoenix DLSS5 runtime. This is
not the full-x64-library route (#2), a helper route (#4), or portable Vulkan.

Verified at 640 x 384, 1:1, on the installed RTX 5080 driver:

- Both x86 and x64 hosts authenticate all 41 model kernels and build the same
  154-dispatch graph per frame using the existing production CPU contracts.
- Six frames complete in one session: three default resets, changed input,
  changed strengths on that same input, and restored original input/defaults.
- All six outputs are byte-identical between the x86 and x64 host controls.
  Default/restored frames also match the separate normal x64 Phoenix DLSS5 API.
- Input and strength changes each produce a different output without restarting
  the session; restoring defaults reproduces the original output exactly.
- GPU outputs are poisoned with half NaNs before each evaluation. Returned
  components must be finite and RGB must differ from the input.
- The host and driver use the same PID, with `host_bits:32`, `driver_bits:64`,
  no helper, and no x64 Phoenix DLSS5 renderer loaded. There are 924 real dispatches.
- All 41 functions and 41 modules are destroyed with zero reported cleanup
  failures. Native compatibility metadata is restored and checked.

The default RGBA16F output SHA-256 is
`01b518eb3040060849e968bb1d5ce5a65764030924ae2b0a7956ae4b58cd89e7`.
CPU tests pin the packet size/offsets in both architectures and reject argument
overflow, pointer truncation, invalid shapes and wrong packet versions.

```powershell
python tools/probe_wow64_tls.py --wow64ext PATH_TO_WOW64EXT `
  --run-gpu-probes --try-d3d12-frame --try-x86-host `
  --ngx-sdk PATH_TO_NGX_HEADERS --ngx-core PATH_TO_INSTALLED_X64_NGX_CORE `
  --large-address-aware --bridge-create-dc --bridge-display-gdi `
  --native-appcompat-view --retry-native-reservation --cuda-first `
  --case thread_driver_host
```

The runner adds matching x64-host and normal x64 Phoenix DLSS5 controls automatically.
The compact receipt is `artifacts/current/wow64-x86-host-investigation.json`;
`x86_host_conclusions` distinguishes full graph execution, cross-host exact
output, and comparison with the native API. A crash, helper, x64 renderer,
incomplete frame, unchanged controls or failed teardown cannot pass.

This establishes **route 3 inference in the isolated host**, not an installed
game route or a complete architecture-neutral public SDK. Current coverage is
reset-only SDR frames and tested strengths; temporal motion/history, resize,
device loss, cancellation and engine capture/presentation still need integration
and validation. The six-frame GPU timestamps are diagnostics; a separate warm
benchmark is documented below. The default helper, add-ons and installed games are
unchanged; nothing is automatically routed through this experimental host.

## All Four Methods

Fresh measurements on RTX 5080 / driver 616.64, 1920 x 1080, 1:1, five-second
warm-up and twenty-second measurement per case:

| Backend | Method | Mean GPU time | Mean call / processing round-trip |
| --- | --- | --- | --- |
| Phoenix DLSS5 | #1: native x64 | 4.007 ms | 4.391 ms |
| Phoenix DLSS5 | #2: whole x64 runtime inside x86 | 4.037 ms | 4.508 ms |
| Phoenix DLSS5 | #3: x86 host / same-process native driver | 4.009 ms | 4.464 ms |
| Phoenix DLSS5 | #4: x86 capture / external x64 helper | 4.338 ms | 12.358 ms |
| NVIDIA DLSS5 | #1: native x64 | 4.027 ms | 4.455 ms |
| NVIDIA DLSS5 | #2: whole x64 runtime inside x86 | 4.004 ms | 4.465 ms |
| NVIDIA DLSS5 | #3: x86 model host | Not available | No x86 NVIDIA DLSS5 model host implementation/source |
| NVIDIA DLSS5 | #4: x86 capture / external x64 helper | 4.472 ms | 13.071 ms |

These are **not game FPS** or identical end-to-end presentation benchmarks.
Methods 1/2 measure isolated GPU-resident native API calls; method 2 enters its
native worker once, so its per-frame x86 handoff is not measured. Method 3
includes a real per-frame x86/native request-response, cached dispatch packets,
GPU execution and completion. It retains input/output textures on the GPU during
timing, includes an output copy, and validates the reset-frame output afterwards.
Its x86 run completed 4,480 measured frames, about 224 completed outputs/second.

Method 4 runs the actual x86 DX12 fixture with the retained external helper and
readback transport. Its wall column is sampled processing round-trip, including
readback/IPC/return, not just a model call. It uses the fixture image rather than
the synthetic input of methods 1-3. GPU timing scopes also differ slightly:
Phoenix DLSS5 native timestamps include output copy, while the helper exposes model
and route times separately. This supports broad processing/transport comparison,
not attribution of small differences solely to process architecture.

Route 3's matching x64 packet-host control measured 4.001 ms GPU / 4.348 ms
synchronized call. The x86 host matched its output exactly and still passed
input/strength/restoration and full module teardown checks. The ordinary x64
Phoenix DLSS5 API supplied another independent output control.

For route-3 timing, add `--benchmark-x86-host` to the route-3 command above.
The current receipt is `artifacts/current/wow64-x86-host-benchmark.json`.
Its CPU packet ABI is versioned and rejects invalid transfer flags. Only the
timed loop may skip repeated host uploads/readbacks; pre/post validation still
poisons GPU output and checks actual finite, changed pixels.

The fresh helper cases were run with:

```powershell
python tools/benchmark_dx12_routes.py --suite PATH_TO_TEST_SUITE `
  --run-gpu --only helper_x86 nvidia_x86
```

`tools/compare_native_methods.py` validates the retained reports and writes
`artifacts/current/native-method-comparison.json`, including source-report
SHA-256 values and per-row timing scope. It refuses failed/mismatched profiles
and marks NVIDIA DLSS5 route 3 unavailable instead of inventing a result. No production
game integration or deployment was changed for this benchmark.

## Earlier Results Before The Reservation Fix

The latest same-process experiment runs both real optimized x64 libraries from a
dedicated native thread inside a disposable GUI-subsystem x86 executable. It
does not launch a helper for inference. Each backend has a separate normal x64
positive control using the same input and model, at 640 x 384 and 1:1 resolution.

| Stage | Normal x64 controls | x64 code inside x86 |
| --- | --- | --- |
| D3D12 device and command queue | Pass | Pass after process-local UMD selection |
| GPU upload/copy/fence/readback | 1,024 exact words | 1,024 exact words |
| Reserve CPU virtual address above 4 GB | Pass | Pass: 64 GiB address, then released |
| Direct CUDA initialization | Success | `CUDA_ERROR_OUT_OF_MEMORY` (2) |
| Legacy NVAPI cubin shader creation/destruction | Pass; creation only, not inference | `NVAPI_ERROR` (-1); no shader |
| Optimized Phoenix DLSS5 session/model | Frame processed; finite changed RGB | Session creation returns `phoenix NVAPI failure` |
| NVIDIA DLSS5 parameter allocation and Init | Success | Success |
| NVIDIA DLSS5 feature creation | Success | `0xBAD00002` (`PlatformError`) |
| NVIDIA DLSS5 evaluation/readback | Three evaluations; finite changed RGB | Not reached |

NVIDIA DLSS5 output is initialized to half NaNs before evaluation. A successful API
return without real finite output writes cannot pass. Phoenix DLSS5 host output is
also initialized to half NaNs. Full-model success requires a passing normal
x64 control, successful process completion, verified GPU transfer and changed
finite RGB samples; DLL loading or initialization alone never counts.

The disposable x86 executable was also tested with `/LARGEADDRESSAWARE`. This
gets Phoenix DLSS5 past an earlier `bad allocation`, but does not fix kernel/model
creation. A native GDI `DeleteDC` export bridge routes that operation to the
same process's x86 coordinator, fixing crashes during native NVAPI/NVIDIA DLSS5
initialization. It performs the real deletion rather than pretending success.
All native objects remain on the native worker; no x86 D3D12 COM object is used
as an x64 interface.

Process-local driver-import probes record selected CUDA virtual allocations,
Windows GPU call statuses and device-control metadata without modifying their
results. A process-identity query experiment did not make CUDA initialize.
The above-4-GB reservation succeeds, so the reported CUDA error does not establish
a simple 4-GB ceiling or actual VRAM exhaustion. The exact driver-internal reason
is still unresolved. These shims are research-only and are not installed in an
add-on, game or system library.

The portable x64 Vulkan alternative also remains unsuccessful. Its earlier
WinTypes fail-fast has been narrowed to an inappropriate WOW64 filesystem-
redirection call from native x64 code. The process-local native identity view
gets beyond that failure in debugger runs. Real `vkCreateInstance` succeeds, but
enumeration through the loader returns `VK_ERROR_INITIALIZATION_FAILED` and no
devices. Calling NVIDIA's x64 ICD directly, with interface version 7 negotiated,
also returns zero devices. Querying the ICD by the actual NVIDIA DLSS5 DXGI adapter
LUIDs does not recover a physical device. This is not a UUID-selection mismatch.

The normal, non-debugged child originally crashed with access violation
`0xC0000005` while debugger runs could return the model error
`no physical device (0)`. A native first-chance observer identified execution
inside **x86 `apphelp.dll`** from the native x64 worker: machine `0x14c`, fault
RVA `0x66c26`, reading x86 absolute-address code as x64 RIP-relative code. This
was not a neural kernel crash and was not resolved by larger address space.

The opt-in `--native-appcompat-view` experiment temporarily separates the native
PEB's compatibility flags/metadata from the inherited x86 shim state. Offsets
come from the current matching ntdll symbols and are bounds/alignment checked;
the image timestamp and distinct WOW64/native PEBs are verified before writing.
The x86 PEB is not changed. Original native values are restored and checked at
normal exit. This avoids the observed foreign-code crash in repeated probes,
but still returns the real `no physical device (0)` model error. It is an
unsupported process-local research step, not a system compatibility or security
setting change and not a successful inference route.

`--compare-appcompat` retains three A/B controls without the separation and two
fresh separated children in one receipt. `appcompat_comparison` explicitly
distinguishes reproducing the x86-apphelp fault, repeated clean error exits and
neural inference (which remains false). A clean error is not a working model.
The normal x64 control using the same portable runtime completes three real
frames. Debugger receipts remain separate and cannot override a failed normal
process result.

In the latest repeated A/B check, one of three unseparated children reproduced
the x86-apphelp fault; the other two returned the model error. Both separated
children returned the model error without an observed fault and verified native
metadata restoration. This supports the crash guard for the observed fault,
not a broad stability claim, driver compatibility or neural inference success.

The external x64 helper remains unchanged and is still the working x86 route for
both optimized Phoenix DLSS5 and NVIDIA DLSS5. Matching-bitness portable x86 Phoenix DLSS5 remains
a separate, slower mechanism; neither is evidence that mechanism #2 works.

Reproduce the latest isolated experiment using external SDK/driver paths:

```powershell
python tools/probe_wow64_tls.py --wow64ext PATH_TO_WOW64EXT `
  --run-gpu-probes --try-d3d12-frame --ngx-sdk PATH_TO_NGX_HEADERS `
  --ngx-core PATH_TO_INSTALLED_X64_NGX_CORE --large-address-aware `
  --try-native-identity --case thread_d3d12_cuda `
  --case thread_d3d12_phoenix --case thread_d3d12_nvidia
```

`artifacts/current/wow64-d3d12-investigation.json` is the compact local receipt;
`d3d12_conclusions` separates normal controls from same-process inference.
Temporary executables/DLLs are not part of the distributable. The probe runner's
zero exit code means the experiment completed with a valid native TLS control,
not that same-process model inference succeeded. CPU-only classification tests:

```powershell
python -m unittest discover -s tools -p "test_wow64*.py"
```

NVIDIA DLSS5 documents that normal 32-bit CUDA applications are unsupported on
[RTX 50-series/Blackwell](https://nvidia.custhelp.com/app/answers/detail/a_id/5615).
That is relevant context, not proof of the cause of this particular x64-in-x86
experiment's failures.

### Additional Driver And Loader Checks

The following bounded variants were tried without changing any installed game,
driver files, global environment variables or persistent registry settings:

| Variant | Result |
| --- | --- |
| Observe real heap, local allocations and file mappings | No observed allocation/mapping failure explaining CUDA error 2 |
| Observe native driver worker-thread creation and entry | Workers start; this alone does not prove correct driver initialization |
| Allocate genuine native per-thread TLS before driver callbacks | Allocation succeeds, but CUDA/model creation still fails |
| Native 64-bit registry view for driver reads | Does not unblock either optimized model |
| Documented lazy CUDA preload and disabled JIT controls | Does not unblock `cuInit` |
| Initialize CUDA before any D3D12 device | Same CUDA error 2 |
| Mapped real x64 main image for driver metadata queries | Does not unblock CUDA/model creation |
| Temporarily use that image in native PEB only | Same failure; original image pointer restored; x86 PEB untouched |
| Separate native PEB compatibility metadata from x86 shim state | Avoids the observed x86 apphelp crash; no GPU/model success |
| Early native IsWow64Process/IsWow64Process2 export view | Does not unblock optimized models |
| Same-process x86 CreateDCA/GetDeviceCaps/GetDC/ReleaseDC/EnumDisplayDevicesA bridge | Calls use real Windows APIs; no model/device-discovery fix |
| Observe dynamic TLS/FLS operations and adapter-query responses | Observed operations succeed; driver-internal rejection remains unresolved |
| Legacy cubin shader API instead of CreateCuModule | Native x64 control succeeds; cross-mode call returns -1 |
| Direct x64 Vulkan ICD, then adapter-LUID enumeration | Instance succeeds; zero physical devices |

All hook slots and temporary native PEB fields are restored on ordinary completion; on a fatal exception the
disposable process exits. No kernel driver, security bypass, fabricated success
status, or rewritten opaque driver packet is used. Native export jump pages are
written before becoming read/execute and retained until process exit because
loaded DLLs may cache them. The current receipts record requested variants and
source hashes, not a claim that these are supported Windows APIs.

Thread callback observations previously showed reused TLS-vector addresses.
That is **not evidence of two live threads sharing a vector**: sequential thread
exit and allocator reuse can produce the same observation. Likewise a callback
that invokes `ExitThread` can bypass our wrapper's return marker; `STILL_ACTIVE`
in that marker is not a reliable post-run liveness result.

Coordinator request layouts are compiled and checked as x86 and x64 before GPU
cases. Requests reject unknown presence flags, unterminated strings and invalid
DEVMODE lengths. CPU-only result tests require normal controls and real changed
neural outputs; the legacy shader result explicitly says `shader_creation_only`.

Additional command options are opt-in research controls, not installer settings:

```powershell
# Add to the D3D12 probe command above as required:
# --bridge-create-dc --bridge-display-gdi --native-registry
# --early-native-identity --native-main-image --native-peb-image
# --native-appcompat-view
# --wrap-driver-threads --repair-driver-tls --cuda-first
# --cuda-no-preload --cuda-no-jit
# Add --case thread_d3d12_legacy for creation-only legacy shader comparison.

# Portable x64 model control, loader/driver discovery and actual cross-mode attempt:
python tools/probe_wow64_tls.py --wow64ext PATH_TO_WOW64EXT `
  --run-gpu-probes --try-portable-frame --vulkan-headers PATH_TO_VULKAN_HEADERS `
  --vulkan-icd PATH_TO_INSTALLED_X64_ICD_JSON --native-registry `
  --try-native-identity --early-native-identity --bridge-create-dc `
  --bridge-display-gdi --direct-vulkan-icd --large-address-aware `
  --native-appcompat-view --compare-appcompat --case thread_frame
```

Without `--direct-vulkan-icd`, the instance/device diagnostic goes through the
standard native loader. Direct ICD calls are diagnostic only: model inference
still uses the original portable runtime and normal Vulkan API. Refer to
[Khronos' loader/driver interface](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md)
for negotiation and adapter-LUID enumeration. CUDA environment controls are
documented in [NVIDIA's initialization API](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-driver-api/group__CUDA__INITIALIZE.html).

## Local Checkpoint

Before this experiment, local `codex/pre-wow64-checkpoint` commits captured:

| Repository | Commit |
| --- | --- |
| Installer/application | `c021ffb` |
| Test runner/local dashboard source | `d909d34` |
| Phoenix DLSS5 add-on | `efedd06` |

Generated runs, logs, payload/model caches and machine-specific cleanup files
were excluded. The test/dashboard checkpoint is local only and does not change
the public source-only branch. Nothing was pushed. Experiment files are separate
on `codex/wow64-native-probe` in the Phoenix DLSS5 checkout.

## Method

The x86 executable dynamically loads an independently compiled WOW64Ext DLL and
uses its `X64Call`, `GetModuleHandle64` and `GetProcAddress64` entry points. Calls
remain inside the probe's own process. It first compares the result of x86 and
x64 `RtlComputeCrc32` calls on identical input.

An import-free x64 control DLL confirms that this is actual x64 DLL code running
inside the x86 process, not just a successful pointer lookup. A separate x64
graphics DLL creates a native D3D12 device/queue and attempts to create/destroy
one verified 53,288-byte kernel module. A normal x64 executable runs the same DLL
as the positive control; no 32-bit game COM pointers are passed into x64 code.

Each case runs in a fresh, bounded child process. Native-thread cases use
`NtCreateThreadEx` to start an import-free x64 entry point in the same process.
The additional flag-2 case skips DLL-attach notifications; it does not hide the
thread from a debugger or disable security features.

An additional variant follows BlackBone's native activation-context setup:
allocate a fresh activation stack, temporarily install it in the native TEB,
invoke the loader, restore the original pointer, and free the stack on normal
return. The private TEB activation-stack offset was checked against this
machine's native ntdll symbols; it is not treated as a portable Windows ABI.
BlackBone was inspected as external source only. No driver was built/installed.

## Observed Results

| Case | Observed result |
| --- | --- |
| x86 calls x64 NTDLL function | Success; CRC matches x86 result |
| x86 loads and calls import-free x64 DLL | Success; control marker returned |
| Normal x64 executable runs graphics worker | Success; capability supported and kernel module created/destroyed |
| Console-subsystem x86 probe loads x64 `kernel32.dll` | `0xC0000142` (`STATUS_DLL_INIT_FAILED`) |
| Console probe loads x64 `kernelbase.dll`, `nvapi64.dll`, `d3d12.dll`, or graphics worker | Same initialization failure |
| Native x64 thread, normal attach behavior | Process access violation `0xC0000005`; no completed loader result |
| Native x64 thread, skip DLL attach | Thread runs and returns the same `0xC0000142` loader failure |
| Fresh activation stack on the calling thread | Allocation succeeds; graphics loader still returns `0xC0000142` |
| Fresh activation stack on the skip-attach native thread | Same result |
| GUI-subsystem x86 probe loads x64 `kernel32.dll` | Success |
| GUI-subsystem x86 probe loads x64 `d3d12.dll` | Success; no device creation demonstrated |
| GUI-subsystem x86 probe loads `nvapi64.dll` | Loads, but `NvAPI_Initialize` crashes |
| GUI-subsystem x86 probe loads graphics worker | Crashes while initializing dependencies, before entering the worker |

### Debugger Diagnosis

The console failure is not a universal rejection of x64 DLLs. Native
`KERNELBASE!ConsoleInitialize` returns false; changing only the probe's executable
subsystem to GUI gets past this obstacle. This is an important correction to
interpreting the first batch of loader failures.

The GUI worker and GUI NVAPI probe then fail in the same native dependency:
`gdi32full!_dyn_tls_init+0x2d`, reached through
`USER32!UserClientDllInitialize`. Its preceding instructions read the native
thread-local-storage vector from `gs:[0x58]`; that vector is null, and the indexed
load faults at address zero. The graphics-worker stack reaches this through
DXGI's `g_DXGKernel` static initializer. The separate NVAPI initialization trace
also reaches native USER32/GDI. This is not evidence that kernel dispatch failed:
the probes have not reached that stage.

The debugger can exit with code zero after observing this fatal fault. The
receipt therefore records the exception independently from debugger exit status.
Symbol-based breakpoints are confined to the disposable process; no persistent
loader flags or registry changes are needed.

The tested approaches cannot replace the x64 helper. A further cross-mode
experiment would need correct native static-TLS allocation/template/index setup
and thread lifecycle handling, then validation of native USER/GDI and graphics
dependencies. Copying an x86 TEB/TLS pointer, supplying a dummy zero-filled TLS
block, skipping DllMain, or suppressing the exception would not establish correct
initialization. No such workaround is integrated into Phoenix DLSS5.

These results do not prove every Windows-internal workaround impossible. They
also do not demonstrate a cross-bitness GPU device/module or neural inference.

## Follow-Up: Real TLS And Native-Thread Probes

The original TLS blocker was narrowed further using a real compiler TLS
template, a nonzero TLS cookie, and a compiler-declared zero-initialized TLS
array. A normal x64 control must validate both before cross-mode results count.
An import-free fixture did not receive normal TLS initialization even in the
x64 control; the corrected fixture imports kernel32 and validates successfully.
The failed fixture is not evidence about the cross-mode route.

The private `LdrpHandleTlsData` routine returns success without allocating native
TLS when the native loader's active-thread count is zero. Instead, invoking
`LdrpInitializeTls` under the native loader lock initializes the actual templates
and indices. Its address is resolved from this machine's matching ntdll symbols,
then checked against the loaded image timestamp and bounds. No fixed Windows
offset, fabricated TLS vector, persistent loader setting or system-file patch
is used. This is still an unsupported private API, not a deployable contract.

### Results After TLS Initialization

| Isolated case | Result |
| --- | --- |
| x86 caller executes x64 TLS fixture | Template and zero-fill checks pass after native TLS initialization |
| Dedicated x64 thread inside x86 process | Same TLS checks pass |
| x64 DXGI library load in x86 process | Pass; former GDI TLS crash avoided |
| Real x64 Phoenix DLSS5 library load and default controls | Pass; all tested default control fields match |
| Phoenix DLSS5 null-session error on borrowed x86 stack | Unhandled `0xE06D7363` C++ exception; debugger flags include invalid stack |
| Same Phoenix DLSS5 error on dedicated native x64 thread | Pass; returns `-1` and `Null native session`, matching normal x64 control |
| Normal x64 graphics/kernel-module control | Pass; NVIDIA DLSS5 device, queue and verified kernel module created; module destroyed |
| Graphics worker on borrowed x86 stack | Unhandled C++ exception before completion |
| Graphics worker on dedicated native x64 thread | Factory and NVIDIA adapter enumeration succeed; D3D12 device creation returns `0x887A0004` (`DXGI_ERROR_UNSUPPORTED`) |
| Cross-mode kernel dispatch or Phoenix DLSS5 frame inference | Not reached; no success claimed |

The native-thread test runs in the **same x86 process**, not a helper. The
returned stack address falls within that native thread's TEB stack bounds. This
fixes the tested error-handling case, but does not establish arbitrary exception
or thread-lifecycle safety in games.

The graphics trace shows D3D12Core reaching UMD loading and loading NVIDIA's
`nvldumd.dll`. Reading the PE headers confirms that file is x86 (`0x014C`), while
the adjacent `nvldumdx.dll` is x64 (`0x8664`). An entry-point-not-found exception
(`0xC0000139`) appears before device creation returns unsupported. This is a
concrete architecture mismatch in the tested loader path, not a failed neural
kernel or evidence that a Phoenix DLSS5 output frame was produced. Why that path was
chosen, and whether a safe process-local alternative exists, remain open.
No driver files, graphics registry entries, or system DLLs were modified.

**Verdict:** actual x64 Phoenix DLSS5 code and its tested error path can run inside
this x86 probe, but mechanism #2 is **not a working GPU inference route**. Do not
expose it as a selectable working add-on mode or remove the existing helper.
Any further research must solve correct native UMD/device creation, then kernel
dispatch, full-frame validation, synchronization, and thread/device teardown.

The dedicated CPU/optional GPU runner is:

```powershell
python tools/probe_wow64_tls.py --wow64ext PATH_TO_CHECKOUT
# Explicitly authorized GPU initialization/module probes and process-local tracing:
python tools/probe_wow64_tls.py --wow64ext PATH_TO_CHECKOUT --run-gpu-probes --trace-gpu
python -m unittest discover -s tools -p test_wow64_tls.py -v
```

Without `--run-gpu-probes`, this runner does not create a graphics device. The
flag alone attempts initialization and kernel-module creation, not full model
inference. `--try-portable-frame` and `--try-d3d12-frame` explicitly attempt real
models and access the configured external weights. The latest TLS/portable
receipt is `artifacts/current/wow64-tls-investigation.json`; a D3D12 (or combined)
run instead writes `artifacts/current/wow64-d3d12-investigation.json`. Temporary
build files are removed. Read the explicit model conclusions rather than
treating runner exit code zero as a successful graphics result.

## Native x86 Vulkan Alternative

The subsequent standalone runtime implementation and full-model validation are
documented in [Experimental Native x86 Vulkan Runtime](X86_VULKAN.md). The
arithmetic probe below remains a separate, smaller capability experiment.

`tests/native/x86_vulkan_probe.cpp` uses the normal matching-bitness system Vulkan
loader, not WOW64Ext or the x64 driver DLL. It creates a compute device/buffer,
passes a 64-bit GPU buffer address to a SPIR-V shader, dispatches 4,096 arithmetic
operations, waits for a fence and checks every returned value. The same source
and shader run as x86 and x64. When available, it additionally dispatches an
independently authored PTX arithmetic kernel through `VK_NV_cuda_kernel_launch`.

On the tested RTX 5080 and installed driver:

| Probe | x86 | x64 |
| --- | --- | --- |
| Vulkan device creation | Pass | Pass |
| 64-bit GPU buffer addresses and integer capability | Present | Present |
| SPIR-V compute/readback | 4,096/4,096 exact | 4,096/4,096 exact |
| `VK_NV_cuda_kernel_launch` advertised | No | Yes |
| CUDA-extension feature enabled | Not available | Yes |
| PTX compute/readback through Vulkan | Not run: extension absent | 4,096/4,096 exact |

Therefore **native x86 GPU compute is possible**, but this driver's x86 Vulkan
interface does not expose the shortcut for reusing our optimized CUDA kernels.
The extension is provisional in the Khronos specification; availability must be
queried, not assumed. An absent x86 extension must not be force-enabled.

### Porting Gaps Found By The Initial Probe

- `native_runtime/backend_vulkan.cpp` encoded allocation IDs above bit 40 in a
  `void*`, then recovers them through `uintptr_t`. On x86 the encoding truncates,
  and shifting a 32-bit `uintptr_t` by 40 is itself invalid. Shader/device support
  does not fix these host-side descriptors. The alternate implementation now
  preserves them in a full-width x86 operand type.
- `Allocation.address`, graph tensor arithmetic, scratch planning and
  `phx::Operation` operands also used pointer-sized storage. The alternate now
  preserves these logical addresses end-to-end, with checked host-pointer
  conversion and 96-byte operator layout tests on both architectures.
- The established Windows Vulkan build script hardcodes x64 compilation/import
  library generation. A native x86 build must include a matching-bitness provider
  and audit exported C calling conventions/decorations and structure layouts.
  The separate alternate validation builder now does this without changing the
  established x64 build script or helper package.
- Game capture/output integration, synchronization, device reset, memory pressure
  inside a 32-bit game's address space remain untested. Standalone whole-model
  evidence is recorded separately in the alternate runtime document.
- The Vulkan arithmetic probe provides no Phoenix DLSS5 frame-time or quality result.
  The portable shader path is not established as a performance replacement for
  the optimized native-kernel helper.

## Recommended Next Work

| Option | Evidence | Recommendation |
| --- | --- | --- |
| Existing x86 capture + x64 helper | Current working route; prior background-delivery results are separate | Keep as the default and fallback |
| Native x86 Vulkan/SPIR-V Phoenix DLSS5 | Arithmetic probe verified; subsequent standalone model port tracked separately | Optional in-process route; do not replace helper routing |
| x86 Vulkan + existing CUDA kernels | Extension absent on the tested x86 driver | Do not enable; revisit only if capability changes |
| WOW64 native x64 graphics | Corrected UMD selection and GDI cleanup permit real GPU copies; Phoenix DLSS5/NVIDIA DLSS5 model creation still fails | Research only, not a game deployment fix |

For an x86 Vulkan port, use this acceptance order:

1. Add 32/64-bit descriptor-width, offset/overflow and ABI regression tests.
2. Build the runtime/provider as x86 and run individual real model operations.
3. Run a full frame against the established portable reference, then temporal
   sequences, unusual sizes and memory-lifetime tests.
4. Integrate in the add-on behind explicit capability/failure fallback.
5. Validate actual game output and processed delivery rate before changing the
   installer default. Do not present game-loop FPS as inference throughput.

## Reproduce

Use an unmodified external checkout of
[WOW64Ext](https://github.com/rwfpl/rewolf-wow64ext) at
`82045488f48c8641609d9a5d8bb832937b9b5acd`:

```powershell
python tools/probe_wow64_native.py --wow64ext PATH_TO_CHECKOUT --run-gpu-probes
# Optional process-local console, graphics and NVAPI initialization traces:
python tools/probe_wow64_native.py --wow64ext PATH_TO_CHECKOUT --run-gpu-probes `
  --loader-debug --debug-case console-loader --debug-case gui-worker --debug-case gui-nvapi
# Matching-bitness Vulkan compute; local Khronos headers and glslang only:
python tools/probe_x86_vulkan.py --deps PATH_TO_VULKAN_DEPS
python -m unittest discover -s tools -p test_wow64_probe.py -v
```

Without the explicit GPU-probe flag, the runner only checks mode switching.
Compiler outputs, DLLs and executables are built in one temporary directory and
removed on completion or failure. One compact current report remains at
`artifacts/current/wow64-native-investigation.json`, outside the distributable
package. The Vulkan probe retains its own compact
`artifacts/current/x86-vulkan-investigation.json`. A zero runner exit code means the experiment completed, not that its
graphics attempt succeeded; inspect the report's explicit conclusion fields.
The WOW64 `--case` option can restrict a run to named cases; a restricted receipt
contains only those cases and is not evidence that omitted cases passed.

WOW64Ext is LGPL-3.0-or-later. Its source remains in the external checkout and its
DLL is dynamically loaded only by this temporary probe. No WOW64Ext source or
binary was added to the Phoenix DLSS5 runtime or shipped package. Any future product
integration needs a separate license review.

References:

- [WOW64Ext documentation](https://github.com/rwfpl/rewolf-wow64ext/blob/master/doc/wow64ext.txt)
- [Microsoft process interoperability](https://learn.microsoft.com/en-us/windows/win32/winprog64/process-interoperability)
- [Microsoft WOW64 implementation details](https://learn.microsoft.com/en-us/windows/win32/winprog64/wow64-implementation-details)
- [BlackBone source and MIT license](https://github.com/DarthTon/Blackbone)
- [Khronos CUDA-kernel launch extension](https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_cuda_kernel_launch.html)
- [Khronos Vulkan headers](https://github.com/KhronosGroup/Vulkan-Headers)
- [Khronos glslang](https://github.com/KhronosGroup/glslang)
