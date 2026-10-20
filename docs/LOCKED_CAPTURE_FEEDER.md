# Immutable captured inputs and worker retries

The optional `CnrPHOENIXFeederEvaluateLocked` export adds a `uint64_t`
capture token to the existing V1 evaluate arguments. Discover it with
`GetProcAddress`; the V1 struct layout, reserved fields and version stay unchanged.

- A nonzero token identifies one immutable captured input. Keep its color,
  depth and motion unchanged until processing finishes. Use a different token
  before supplying another input, including another eye's captured image.
- Seed the output from that original input before each retry. Submit and retire
  every recorded GPU command list before checking completion. Deferred,
  unsubmitted attempts do not consume the input's pending history reset.
- A recorded call is not completed neural output. The helper checks current
  routing and actual worker delivery/composition counters after its queue fence
  retires. It never accepts a passthrough merely because a call was recorded.
- A worker result for this same locked input does not expire while its CPU/GPU
  computation runs. Results from other tokens or generations are rejected.
  Ordinary engine frames and V1 calls retain their configured frame/time limits.
- The helper waits at most 55 seconds for a reference CPU request and 15 seconds
  otherwise. This supports bounded slow reference processing; it does not make
  CPU execution suitable for real-time gameplay.

The scope is thread-local and restored when evaluate returns. It does not change
the portable runner protocol, model layout or native RTX kernel implementation.
