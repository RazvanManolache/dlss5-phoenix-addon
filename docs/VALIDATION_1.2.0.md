# Phoenix 1.2.0 validation

- The DX9 x86 Museum fixture delivered fresh single-sample game depth during camera motion.
- On an RTX 5080, a Direct3D 9 probe resolved 4× multisampled depth into an R32F texture and recovered the expected values. Live DX9 gameplay then showed world geometry in the depth preview and updated it when the camera moved, including after a fullscreen reset and gameplay resume.
- Live helper telemetry showed completed neural inference with game depth selected and Depth Only disabled for both Phoenix and NVIDIA DLSS5.
- Windows release validation covers CPU contracts, software D3D12 rendering, both addon architectures, DLL loading, and embedded payload identity and extraction. The Linux release workflow builds and tests the Vulkan layer and native SDK.

The live depth checks cover one NVIDIA GPU. Older RTX architecture packages retain their experimental status. Depth estimators and neural rendering models use separate files.
