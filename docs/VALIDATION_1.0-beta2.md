# Phoenix DLSS5 1.0 Beta 2 validation

The release's `VALIDATION.json` binds the shipped files to the source commit and completed checks. `SHA256SUMS.txt` identifies the downloads. A compiled backend, isolated check, live-game observation and full compatibility qualification have different scopes.

## Publication checks

- Compile matching x86/x64 addons and graphics helpers from current source; exercise CPU contracts and model-free software D3D12 capture, local-surface, presentation and motion contracts.
- Check actual addon load/unload, required exports, resource extraction and nested x64 helper/core identity.
- Exercise provider retirement while a worker still holds its callbacks, and reject stale runtime guide ownership during recreation. Targeted CPU/software-renderer regression checks cover these lifetimes independently of game testing.
- Check the driver-discovery, validation-cache and compact-pixel contracts without loading a hardware driver or neural model.
- Validate the optional native SDK's additive output-postprocess contract with software D3D12. Bind updated runtime files to their actual build receipts; retain the other runtime/kernel payloads by hash.
- Verify the separate optional input companion's source/binary bindings and its documented load/mailbox/session checks.
- Audit the complete committed source and recursively inspect distributed embedded payloads for excluded models, private paths, credentials, captures and logs.
- Compare source-archive blobs with the publication commit, verify archive contents against file inventories, and verify public downloads against their checksums.

Counts, outcomes and hashes are recorded after the checks run; this checklist alone is not a success receipt. A reused binary retains its original validation scope rather than acquiring fresh inference claims through repackaging.

## Evidence boundaries

No games or hardware-GPU inference are run as part of this publication. Previously measured model timings and memory numbers keep their dates and fixture identities. The frame-delivery guide distinguishes modest isolated worker changes from actual-game delivery rates.

The experimental x86 bridge's installed-driver discovery removes a fixed CUDA-driver path; it does not remove the remaining Windows bootstrap compatibility checks or qualify every driver. A failed driver/profile check retains helper fallback.

RTX 20/30/40, Metal, HIP and SYCL do not gain new physical-device qualification from these packages. Hidden-desktop game testing is not included as a user-facing runtime promise.

[Execution mechanisms](EXECUTION_MECHANISMS.md) · [Frame-delivery study](FRAME_DELIVERY_OPTIMIZATION.md) · [Performance and memory](PERFORMANCE.md) · [Support matrix](SUPPORT.md)
