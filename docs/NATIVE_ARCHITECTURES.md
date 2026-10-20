# Native RTX architecture variants

The current source builds architecture-specific Phoenix kernels for both
**Original** and **Lightweight FP8**. This is separate from choosing a community
NVIDIA runtime: these kernels execute through Phoenix's own Native RTX backend.
Model resources, packed storage, graph topology and the 41-module launch ABI stay
the same. Native model folders and supported NVIDIA DLL weight sources remain
available.

| Target | Consumer RTX generation | Arithmetic / memory path | Validation boundary |
| --- | --- | --- | --- |
| `sm_75` | RTX 20 | Packed E4M3 storage decoded to FP16 Tensor Core operations, split into k8 steps; synchronous shared copies | Experimental; compile/offline checks, no device inference |
| `sm_86` | RTX 30 | Packed E4M3 decoded to paired k16 FP16 operations; synchronous bulk copies and supported word-copy instructions | Experimental; compile/offline checks, no device inference |
| `sm_89` | RTX 40 | Native FP8 matrix arithmetic; synchronous bulk-copy fallback and supported word-copy instructions | Experimental; compile/offline checks, no device inference |
| `sm_120` | RTX 50 | Existing FP8 Tensor Core, bulk-copy/barrier and vector-atomic kernels | All 41 images byte-identical to the SDK baseline; combined modules passed RTX 5080 reset/history and timing comparisons for both models at 320×180 and 1080p |

The driver selects an architecture image when creating each NVAPI module; no
per-frame CPU/GPU architecture branch is added. Only the chosen code executes.
The multi-architecture binaries contain cubins, not fallback PTX that could be
silently JIT-compiled for an unsupported GPU.

## Automatic selection in the experimental addon

Phoenix 0.12 publishes this path as the separate **RTX Compatibility Windows**
download, for both x86 and x64. The standard Windows download keeps the validated
single-target RTX 50 payload. See [package selection](RELEASE_0.12.md); install
only one variant. The stable embedding gate is not relaxed by the public experiment.

Select Phoenix with the Native RTX executor as usual; there is no separate
older-card switch. Each packaged module contains all four targets. Phoenix
passes the module and the active D3D12 device to NVAPI, and the driver selects
the compatible image for that device when creating the module:

- RTX 20 uses `sm_75` compatibility kernels.
- RTX 30 uses `sm_86` compatibility kernels.
- RTX 40 uses `sm_89` kernels.
- RTX 50 uses the preserved `sm_120` kernels.

Both Original and Lightweight use this same automatic loading path. Selection
uses the processing device rather than the first GPU name found on the machine,
so a helper processing on another adapter uses that adapter's compatible image.
Phoenix does not redirect the NVIDIA DLSS5 backend to its own kernels; that
backend continues to execute the selected NVIDIA runtime.

Pending older-card validation does not disable these paths in the explicit
experimental package. Unsupported hardware or a driver that rejects module
loading still reports a startup failure through the existing route handling;
Phoenix does not force a mismatched architecture image onto the GPU.

## Arithmetic and synchronization

`common/architecture_math.cuh` supplies the pre-Ada conversion and matrix helpers.
Small integer conversions implement saturating E4M3 with nearest-even rounding,
signed zero and NaN handling, preserving the packed two-byte interface. An FP8
k32 operation is decomposed into two FP16 k16 operations; Turing further splits
each into two k8 operations. This follows OpenNR's compatibility approach, with
attribution retained in [THIRD_PARTY.md](../THIRD_PARTY.md). It does not retrain or
expand the installed model weights.

Older targets use the existing synchronous bulk-copy branches and their matching
block barriers. Turing's explicit word-copy stage also uses synchronous loads and
shared stores. Four packed-half atomic updates replace unsupported vector atomics;
they remain atomic and retain their address/update order. RTX 50 keeps the original
instruction paths. The fallback work can cost speed, and altered accumulation can
affect output rounding on older hardware; no pixel-parity or performance claim is
made for those targets before device testing.

## Building

The production builder defaults to all four architectures. CUDA 13.1 and the
existing MSVC setup were used for the offline checks. Build scripts use Python;
the deployed addon, kernels and model processing do not.

```powershell
python tools/production_build_native_kernels.py native_arch_build `
  --compiler 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe' `
  --jobs 4 --preserve-sm120 artifacts/current/sdk/windows-x64/native
```

`--preserve-sm120` verifies each baseline graph/launch contract and all 41 cubin
hashes before copying those RTX 50 binaries unchanged. Missing, altered or
incompatible baselines are rejected. Omit it to compile all targets from source;
use `--architectures sm_120` for the previous single-architecture output format.
Additive model-identity metadata is accepted during graph migration; execution
plans and existing identity fields must still agree.

Large compatibility kernels can take more than ten minutes to compile. The
per-command limit defaults to one hour and can be set with `--compile-timeout`
(seconds). After an interrupted build, repeat the command with `--resume`.
Resuming checks the recorded source/graph snapshot, build recipe and CUDA image
before reusing a completed kernel. Source or graph changes require a fresh build.

Each combined module has all four architecture images and a SHA-256 pin.
`architecture_images` records each image's target, size and hash. The manifest
records `selection_status=compiled`, `validation_pending=true` and per-architecture
pending inference status. Compilation never grants inference validation or bypasses
the stable addon embedding gate. The saved-hash validator handles single-architecture
bundles; one GPU cannot certify the other images. Stable packaging requires every
declared target's device validation state to be `validated`.

An explicit `tools/build_addon.py --experimental-native` build can embed a
successfully compiled, model-free bundle while retaining its pending architecture
validation states. The addon identifies itself as experimental in its exported
name and overlay. The stable default remains strict.

To build and test the experimental Windows x64 package from an existing validated
addon runtime and the new kernels:

```powershell
python tools/build_experimental_rtx_addon.py `
  --source-addon artifacts/current/addon/Phoenix.addon64 `
  --native build/native_arch_build
```

This command performs GPU tests. It writes the current ZIP, standalone addon,
validation receipt and checksums under `artifacts/current/addon-experimental-rtx`.
The installed addon needs no Python, compiler or CUDA SDK. Model files remain
external. Published addon/SDK files are not replaced by these local builds.

## Verification

```powershell
python tools/test_native_architectures.py `
  --bundle build/native_arch_build `
  --baseline artifacts/current/sdk/windows-x64/native `
  --model '../research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll' `
  --output artifacts/current/native-architectures/validation.json
```

This runs CPU/offline checks only: CUDA's binary extractor independently unpacks
all 164 images, verifies their architecture/entry/hash and checks all 41 RTX 50
images against the baseline. An independent grid oracle exercises all 65,536 half
bit patterns and all 256 E4M3 codes through the compatibility converters. Real
model graph/launch tests use injected recording functions instead of NVAPI/GPU
execution. No game, installed configuration or frozen model reference is changed.

An explicitly requested arithmetic device check can compare both compatibility
MMA decompositions with native FP8 on an Ada or newer GPU:

```powershell
python tools/test_native_architecture_arithmetic.py --gpu `
  --output artifacts/current/native-architectures/arithmetic.json
```

The fixed hashed operands use exactly representable products/sums to distinguish
fragment mapping from expected reduction-rounding differences. Both decompositions
match native FP8 in 49 warp cases (1,568 lane results each) on RTX 5080. A deliberately
corrupted operand must fail the same test. This exercises compatibility arithmetic
on the current GPU; it does not validate an older card, its driver or full-model
performance. Omit `--gpu` to run CPU conversion checks only;
`--gpu-architecture sm_89` builds the arithmetic probe for Ada.

## Current measured results

The complete four-target build passed independent extraction checks for all 164
device images. CPU model/plan checks passed 104,448 assertions, including 16
Lightweight dispatch-depth cases, 15 block-skip cases and 12 frame extents.
All 27 focused Python tests passed. Receipts:
[images and plans](current-native-architectures/validation.json),
[arithmetic](current-native-architectures/arithmetic.json),
[paired RTX 5080 comparison](current-native-architectures/paired.json).

| Phoenix model | Extent | Previous GPU ms | Combined-module GPU ms | Output |
| --- | --- | ---: | ---: | --- |
| Original | 320×180 | 1.852 | 1.850 | Exact reset/history match |
| Lightweight FP8 | 320×180 | 1.016 | 1.017 | Exact reset/history match |
| Original | 1920×1080 | 3.843 | 3.837 | Exact reset/history match |
| Lightweight FP8 | 1920×1080 | 2.271 | 2.270 | Exact reset/history match |

These are deterministic synthetic-input fixtures on RTX 5080, not gameplay FPS.
1080p values are medians of three process medians with alternating before/after
order; 320×180 uses one pair. Each process warms up for 0.5 seconds and measures
for one second. Captures also match across repeated processes. Changes below
0.2% are within the observed timing variation, not a speedup claim.

Combined modules occupy 22.67 MiB versus 3.40 MiB for the single-target baseline.
Startup latency and peak host memory were not benchmarked. Older-card full-model
inference, quality and timings remain unvalidated, and the generated bundle
retains its compiled/pending-release-validation status. Published packages and
the validated SDK baseline remain intact.

Before promoting the experimental addon to stable, complete release validation
and test reset/history behavior, numerical/visual results and timings on each older
generation. Driver support for the NVAPI module-loading interface is
also required. These source targets do not certify every GPU sharing a series
name, non-RTX cards, or a particular older driver.

## Packaged addon verification

The local experimental Windows x64 addon passed all 22 CPU suites and both
software D3D12 suites. A minimal ReShade host loaded the actual DLL, checked its
experimental identity and feeder ABI exports, and verified balanced registration
and unloading. The production extraction code verified all 109 embedded files,
including decompression and hashes, in a temporary cache. No NVIDIA model DLL is
included. These checks do not validate retail-game presentation.

Both models then ran on the RTX 5080 using the kernels and forwarding shim
extracted from that actual addon. All 41 embedded `sm_120` images match the SDK
baseline byte for byte. Reset/history captures match exactly at both extents.

| Phoenix model | Extent | Baseline GPU ms | Packaged GPU ms |
| --- | --- | ---: | ---: |
| Original | 320×180 | 1.848800 | 1.847200 |
| Lightweight FP8 | 320×180 | 1.013536 | 1.015808 |
| Original | 1920×1080 | 3.828864 | 3.829632 |
| Lightweight FP8 | 1920×1080 | 2.261312 | 2.262016 |

Each case uses one paired process comparison with 0.5 seconds of warmup and one
second of measurement. These synthetic fixtures show essentially unchanged
timings; they are not gameplay FPS or a statistical speedup claim. Older RTX
20/30/40 device inference remains untested. That development run changed no game
files and uploaded no release. The [packaged-addon receipt](current-native-architectures/addon-validation.json)
records the exact addon hash, suite results, captures and embedded-resource audit.
