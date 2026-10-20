# Native WOW64 GUI Initialization

The execution sweep exposed failures that the original compute-only client did
not exercise. This document describes the local fix; it does not certify retail
games, protected games or arbitrary Windows/driver versions.

## Fresh Worker Activation State

The real addon starts native processing from a fresh x86 worker. The original
isolated client initialized the native loader from its main thread first,
masking missing native per-thread activation state.

The failing worker dereferenced a null native
`_TEB.ActivationContextStackPointer` inside
`ntdll!RtlQueryInformationActivationContext` while loading native KernelBase.
The worker-first isolated client reproduced the same failure without a game.

`NativeActivationContext` now queries the current native TEB, allocates an
activation stack only when absent, and restores the original slot on scope
exit. The bootstrap's existing ntdll hash/timestamp checks run first. The
private TEB offset is not treated as a portable Windows API.

## Mixed GUI Callback Dispatch

Loading native USER32 replaces the process's native PEB callback-table pointer.
The fixture then entered its x86 window procedure with a native code selector.
Simply restoring the WOW64 table protected the game but sent native CUDA/driver
workers through an uninitialized x86 callback context instead.

The experimental bootstrap installs process-owned callback tail thunks. Native
threads with loader-managed native TLS dispatch to the native USER32 table;
ordinary WOW64 threads dispatch to the game's original WOW64 table. The thunks
preserve the callback arguments and stack, use volatile R11, and are made RX
before publication. Their memory and native images stay loaded until process
exit because driver threads can retain callback targets.

This changes process-local callback state; it does not patch a system image or
install a global hook. It remains an experimental, profile-bound mechanism.
The current implementation covers 256 callback slots and relies on the verified
native TLS/TEB layout. It is not a universal cross-Windows compatibility claim.
Native sessions retain their existing single-owner lock.

Native USER32 is loaded and the dispatcher installed before loading native
DXGI/driver dependencies, so those initializers can notify existing game
windows through the correct callback path. USER32 can change the table inside
its initializer, before `LoadLibrary` returns. That initialization uses a
temporary thread-local native PEB view with shared loader/heap pointers; changed
fields are merged back with compare/exchange, excluding the callback table.
The live game therefore keeps its original table until the mixed dispatcher is
ready. The view remains process-owned in case a native dependency retains its
address. Its size and offsets are private, verified for the bound Windows image.

The updated package requires
`phoenix-wow64-bootstrap/2`; an older package is rejected before native loading.
The matching addon and optional runtime must be updated together.

## Legacy SDR Metadata

ReShade returns `color_space::unknown` for legacy DX9, DX10 and OpenGL
swapchains. Treating that as HDR rejected otherwise usable native transports.
The controller now infers SDR for ordinary eight-bit RGB/BGR canvases when
metadata is unknown. Explicit HDR, float formats and unknown ten-bit formats
remain rejected. This inference does not change resolution or neural controls.

## NVIDIA DLSS5 Evidence

The native library path already completed NVIDIA DLSS5 evaluations, but its panel
snapshot omitted feature readiness. The benchmark correctly refused to score
that as verified NVIDIA DLSS5 processing.

The model packet now reports actual feature-creation/evaluation results and
counters after the corresponding successful GPU submissions. The caller checks
the packet size, and the worker publishes these counters through its snapshot.
No helper fallback is counted as a native-library pass.

## Validation Boundaries

- The isolated client now starts on a fresh x86 worker, before main-thread
  priming. It checks actual inference, NVIDIA DLSS5 counters, controls, refinements,
  renderer switching and clean retirement.
- The synthetic fixture sweep additionally checks observed execution/PID,
  sustained delivery, enhanced pixels, a final screenshot and process exit.
  Its current machine-local report is separate from release validation.
- The external helper remains available and remains the automatic x86 default.
- Both native x86 paths still stage pixels through CPU memory. Same-process
  inference does not imply zero-copy transport or faster complete-frame output.
- A compute-only success is not GUI, visible gameplay, temporal, resize or VR
  validation. Successful frozen-scene captures do not prove equal image quality.

Relevant implementation: `wow64/client32.h`, `wow64/bootstrap64.cpp`,
`wow64/model_contract.h`, `wow64/model64.cpp`, `wow64/graph32.h` and
`host_client.h` under `src/integration/compatibility`.
