# Phoenix 1.3.0 validation

- Windows package checks cover x86 and x64 DLL loading, embedded payload extraction, matching helpers, and the core CPU and software D3D12 test suites. The RTX Compatibility package includes experimental kernels for older cards. Physical older-card inference remains unverified.
- A pre-release Windows build was exercised on an RTX 5080 with generated Direct3D, OpenGL, Vulkan, and legacy-wrapper fixtures. Of 160 previously failing rows rerun in the last targeted pass, 156 passed. Four forced WOW64 tests hit the x86 address-space guard on DX10 WARP. Auto and the x64 helper produced processed output on that fixture.
- The broader Windows release regression was still running when these notes were prepared. WARP measures CPU-rendered fixtures at 640×360 and has a separate timing profile. DX10 and DX10.1 WARP can use basic Portable with motion off. Automatic uses the helper when Portable frame services are requested on these renderers.
- The Linux release workflow builds the Vulkan layer and SDK and checks the relocatable package. It runs without physical GPU validation.
- Frame delivery and motion costs depend on graphics API, resolution, model, and selected route. [Motion benchmark methodology](motion-providers.md)
