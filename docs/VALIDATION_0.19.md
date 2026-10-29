# Phoenix DLSS5 0.19 validation

## Fresh release gates

Both x86/x64 final addons are checked for loading, feeder ABI, balanced registration/unload, recursive embedded-payload hashes and extraction. The standard and experimental source builds run 24 CPU suites and four software D3D12 suites. The shared ImGui interaction tests exercise both architectures, including the label toggle outside comparison mode. HUD-only settings do not require a helper restart.

The packaged inference graph, kernel images and portable companions retain their published 0.18 identities. The newly built addon, nested helper core and graphics helper have fresh source-bound identities. No model DLLs, extracted weights, private configuration or logs are shipped.

## Targeted runtime evidence before packaging

The temporal-stability checks exercised both Original and Lightweight in Bodycam with automatic engine-motion selection and forced NVIDIA optical flow. At the recorded checkpoints, 389/659 engine-motion acquisitions completed without submission refusals or outstanding reader leases. Motion-only snapshots are allowed only for consumers that can run without engine depth. Other consumers retain their depth requirements.

These are scoped route/lifetime checks and local observations, not controlled performance benchmarks or all-game visual acceptance. The packaging checks launch no retail games and run no physical-GPU inference. No older-RTX, AMD/Intel or full API-matrix qualification is claimed by this publication. Existing benchmark reports retain their original build and hardware scope.

Source and binary archives are audited for excluded content. Public release assets are downloaded independently and SHA-256 checked.
