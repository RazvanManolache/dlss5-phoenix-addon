# Phoenix 1.1.5 validation

This update reuses the completed functional campaign and targeted regression evidence. It does not claim a new full game-coverage or performance campaign.

- Both x86 and x64 UI suites passed 1,918 widget checks and 2,365 dashboard checks per architecture, including uniform button hit areas, wrapping, model selection and retained FPS histories.
- Control/policy checks passed on both architectures, including a response-file lock, exactly-once command dispatch and recovery after release of the lock.
- The release core builds passed 35 CPU contract suites and 6 software D3D12 suites. Compatibility packaging checked both add-on architectures, embedded resources and exports.
- Appearance regressions covered Original and Lightweight FP8 with intensity above 1 on CPU, native CUDA and native D3D12 paths.
- Live processing verified both models with Phoenix and NVIDIA through the same in-process D3D11/D3D12 texture route. No external helper was required. Fresh installation paths and runtime hashes were checked separately.
- F6 passed a complete live off/on check. A later synthetic-input recheck missed some injected key events and is inconclusive; an input acknowledgement alone is not evidence of a rendered action.
- A live blocked-response/recovery check observed continued completed-frame progress and cleared control-channel diagnostics. The command was not replayed.

The recent same-route FP8 diagnostic medians were 2.566 ms for NVIDIA and 2.581 ms for Phoenix on the tested RTX 5080. These measure neural GPU work, not game FPS or end-to-end latency.

A frozen-image diagnostic showed similar temporal variation in both implementations. Skipping learned blocks increased variation; keep each block and network-pass count at 1 for the trained schedule. Temporal stability remains scene dependent.

Release packaging independently checks source privacy, embedded payload hashes, archive contents and public download checksums. These checks establish package integrity rather than additional physical-device or gameplay coverage.
