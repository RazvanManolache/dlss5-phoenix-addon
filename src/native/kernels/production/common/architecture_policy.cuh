#pragma once
// Instruction strategy is separate from nvcc's physical compilation target.
// Defaults preserve native code selection. Overrides only lower capabilities;
// they do not impersonate an older GPU or change the model/launch ABI.
#ifndef PHOENIX_KERNEL_STYLE
#define PHOENIX_STYLE_ARCH __CUDA_ARCH__
#else
#if PHOENIX_KERNEL_STYLE != 750 && PHOENIX_KERNEL_STYLE != 860 && PHOENIX_KERNEL_STYLE != 890
#error Unsupported Phoenix kernel style
#endif
#if defined(__CUDA_ARCH__) && PHOENIX_KERNEL_STYLE > __CUDA_ARCH__
#error Kernel style requires capabilities newer than the physical target
#endif
#define PHOENIX_STYLE_ARCH PHOENIX_KERNEL_STYLE
#endif
