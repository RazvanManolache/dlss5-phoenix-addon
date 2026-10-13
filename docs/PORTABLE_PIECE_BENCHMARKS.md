# Measuring individual portable operations

`phoenix_cpp_graph_profile.exe` can time one operation or a consecutive block
using operands produced by an actual forward-pass prefix. This helps distinguish
launch overhead, matrix work and row normalization before testing a whole frame.
Use CUDA or Vulkan; this mode requires GPU timestamps.

Build the tool with the requirements in the main README:

```powershell
python tools/build_portable_cpp.py piece_tools --only graph_profile
```

Supply an external runtime, compatible external model DLL and RGB32F input.
The example selects a 1080p graph with 512 windows per chunk. The RGB file must
contain 1920 × 1080 × 3 little-endian float32 values. Device IDs come from the
runtime's device enumeration tool. Nothing extracts a model into the repository.

```json
{
  "width": 1920,
  "height": 1080,
  "padded_width": 1920,
  "padded_height": 1088,
  "config": {
    "runtime_root": "C:/Bench/runtime",
    "backend": "cuda",
    "device": "cuda:GPU-YOUR-DEVICE-UUID",
    "window_batch_limit": 512,
    "cuda_direct_fp8": true,
    "cuda_fused_feed_forward": true,
    "cuda_fused_window_block": true,
    "cuda_shape_matrices": true,
    "weight_source": {
      "kind": "nvidia_dll",
      "path": "C:/Bench/nvngx_dlssnr.dll"
    }
  },
  "cases": [{"color": "C:/Bench/input.rgb32f.bin"}],
  "piece_repetitions": 32,
  "pieces": [
    {"family": 1, "m": 32768, "n": 128, "k": 32, "batch": 1, "span": 9},
    {"family": 1, "m": 640, "n": 4096, "k": 1024, "batch": 1},
    {"family": 2, "kind": 6, "n": 640, "batch": 32}
  ]
}
```

```powershell
./build/piece_tools/phoenix_cpp_graph_profile.exe piece_job.json piece_result.json
```

- A selector matches the first operation with those dimensions. `family`,
  `kind` and `flags` use the [operator ABI](OPERATOR_ABI.md). For matrices,
  `m`, `n`, `k` and `batch` are rows, columns, reduction width and batch count.
- `span` defaults to one. Nine selects the complete eligible 32-channel window
  chain in this example; two selects its feed-forward pair. The result records
  the selected command index and flags, so different operations are not silently
  compared under the same label.
- The prefix uses the first RGB image, default controls and empty history.
  It runs outside the timed interval. Scratch inputs that would otherwise be
  overwritten during replay are preserved in an immutable snapshot.
- Each sample replays the piece 32 times under grouped GPU timing. The tool
  discards two warmups and returns three samples and their median. Transfers,
  snapshot creation and graph capture are excluded from the GPU piece time.
- The snapshot needs one additional scratch arena on the device and a host
  copy. `benchmark_snapshot_bytes` reports it separately. `resident_bytes`
  describes the production graph's owned allocations. **Do not use the piece
  process's memory consumption as a production memory measurement.**
- A successful timing run proves execution, not numerical equivalence.
  Validate complete outputs and temporal sequences separately. Detailed
  per-command profiling disables block fusion; use `pieces` to measure the
  actual fused block.

The optional `cuda_vendor_matrix_benchmark` build target compares synthetic
matrices against cuBLASLt. Running it requires the CUDA Toolkit's cuBLASLt DLL
on the process search path. Its encoding, multiplication and publication times
are reported together; it is not a whole-model benchmark or an addon dependency.
