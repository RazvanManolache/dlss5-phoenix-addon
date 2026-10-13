#pragma once
#include <cstdint>
namespace phoenix_cpp {
// Optional factory flags leave the original backend ABI usable by older packages.
enum BackendOption : uint32_t {
    ScalarMatrices = 1u << 0,
    DispatchProfiling = 1u << 1,
    OrderedAttention = 1u << 2,
    LosslessFp8Matrices = 1u << 3,
    NarrowMatrixWorkgroups = 1u << 4,
    WideMatrixWorkgroups = 1u << 5,
    MatrixColumns32 = 1u << 6,
    MatrixColumns64 = 1u << 7,
    CudaDirectFp8 = 1u << 8,
    Fp8Columns64 = 1u << 9,
    Fp8Columns128 = 1u << 10,
    CudaFusedFeedForward = 1u << 11,
    CudaFusedWideFeedForward = 1u << 12, // Retained 64-channel experiment; measured slower.
    CudaShapeMatrices = 1u << 13,
    CudaFusedWindowBlock = 1u << 14, // Shared-memory 32-channel window block.
};
enum BackendFeature : uint32_t {
    FusedWindowAttention = 1u << 0,
    DeviceFeaturePacking = 1u << 1,
    FusedMatrixResidual = 1u << 2,
    PreferUnfusedWindowAttention = 1u << 3 // Scheduling hint, not missing operation support.
};
}
