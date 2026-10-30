# Installation and package choice

[Project overview](../README.md) · [Documentation index](README.md)

## Start here

1. Close the game and keep a copy of your existing configuration.
2. Install [ReShade with full addon support](https://reshade.me/). A standard build without addon support cannot load Phoenix.
3. Download **one** main Windows package from the [1.0 Beta release](https://github.com/RazvanManolache/dlss5-phoenix-addon/releases/tag/1.0-beta). Use the RTX Compatibility package for its experimental older-RTX kernels; do not install both main addons together.
4. Install the addon matching the **game process**: `Phoenix.addon32` for x86 or `Phoenix.addon64` for x64. Keep the package's helper/runtime files and relative layout together; the [package instructions](RELEASE_1.0-beta.md#downloads) identify their contents. Do not mix helper/addon versions when updating.
5. Supply a compatible model separately, then select and validate its DLL or model folder in **Setup**. See [shared models](SHARED_MODELS.md), [model selection](MULTI_MODEL.md) and [compatibility](COMPATIBILITY.md). Model weights and NVIDIA runtime DLLs are not bundled with this project's downloads.
6. Start with **Automatic** execution, select Phoenix DLSS5 or NVIDIA DLSS5, and use comparison views to check the result. Continue with the [user guide](USER_GUIDE.md).

## Optional components

- **Phoenix Input** is a separate companion for background input and automation; rendering does not require it. [Setup and controls](PHOENIX_INPUT.md).
- **Portable runtimes** are optional experimental backends, with their own hardware/dependency limits. [Support matrix](SUPPORT.md) · [External dependencies](../third_party/EXTERNAL_DEPENDENCIES.md).
- **32-bit games** run on 64-bit Windows and may use the retained x64 helper. [Execution choices and exact limits](COMPATIBILITY.md#32-bit-games-x86-compatibility).

## Package provenance and validation

[Download beta](https://github.com/RazvanManolache/dlss5-phoenix-addon/releases/tag/1.0-beta) ·
[Installation and package choice](RELEASE_1.0-beta.md) · [Controls](USER_GUIDE.md#a-quick-look) ·
[Why I'm making this](PROJECT.md#why-im-making-this) · [Support and limits](SUPPORT.md)

The 1.0 Beta packages include the current addon, graphics helper and model-free kernels. The standard package targets RTX 50; the separate RTX Compatibility package includes compiled RTX 20/30/40/50 images and selectable instruction styles. Physical older-RTX inference remains unvalidated.

Package checks and earlier isolated GPU studies have separate scopes. See [release validation](VALIDATION_1.0-beta.md), [native performance](PERFORMANCE.md), [portable tuning](PORTABLE_REUSE_STUDY.md) and [resolution profiling](NATIVE_SCALING_PROFILE.md).
