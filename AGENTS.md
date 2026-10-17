# Project working instructions

- Keep source, Git history and all uncommitted changes intact when cleaning or moving this checkout. Never reset, force-checkout or discard another task's edits.
- Do not accumulate dated build directories, duplicate release packages, benchmark inputs, screenshot archives or test history. Use temporary build directories, promote only the requested current outputs into `artifacts/current`, then remove the temporary directory on success and failure.
- Keep at most one current addon package, model-free native kernel bundle and SDK for each platform. Keep compact manifests and external model pointers; do not copy model DLLs or large inputs into every build.
- Build outputs are reproducible from the maintained source and documented external SDKs. `third_party/vulkan-loader.def` is a retained build dependency used by the sibling test bench; do not move it back under disposable `build` output.
- Do not launch games, GPU inference, DOSBox, device tests, exhaustive suites or repeated benchmarks without an explicit user request. This cleanup/move does not authorize builds or tests. Never report compiled or copied files as runtime validation.
- Restrict recursive cleanup to verified absolute paths inside this project. Never follow asset junctions or remove their external targets. The obsolete local `examples/gallery_lab` was removed at the user's request; do not recreate it or its external artwork junctions. Use the sibling `phoenix-gallery` and `dlss-everything-testing` projects.
- Installed games and their configuration are outside this task's scope. Preserve the external model, NGX SDK and Vulkan dependencies. Keep `DO_NOT_TOUCH_DLSS_REFERENCE_20260907` read-only and untouched.
- After relocation, update active sibling project configurations; do not rewrite historical reports to claim they ran at the new location.
