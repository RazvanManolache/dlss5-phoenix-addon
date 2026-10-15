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
    VulkanWideColumns32 = 1u << 15,
    VulkanWideColumns64 = 1u << 16,
    CudaCompactWindowBlock = 1u << 17,
    VulkanShapeMatrices = 1u << 18,
    VulkanHardwareHalf = 1u << 19,
    VulkanGenericWide = 1u << 20,
    VulkanTiledFallback = 1u << 21,     // Shared-memory FP32 tiles; no matrix extension required.
    VulkanCoalescedFallback = 1u << 22, // Contiguous loads for measured fallback matrix shapes.
    VulkanScalarAttention = 1u << 23,   // Shared-memory attention without matrix extensions.
    VulkanFusedQkv = 1u << 24,          // Combine eligible local Q/K normalization and V packing.
    VulkanCacheNoise = 1u << 25,   // Retain seeded noise on the tested GPU; release its host copy.
    VulkanWorkgroupFp8 = 1u << 26, // Capability-checked workgroup/tensor-addressed FP8 matrices.
    VulkanFusedFeedForward = 1u << 27, // Register-resident 32->128->32 feed-forward pair.
    VulkanProjectedQkv = 1u << 28,     // Project and normalize eligible 32-channel Q/K/V together.
    VulkanWorkgroupShapeTiles = 1u << 29, // Select 32/64-row tiles by the measured matrix shape.
    VulkanWorkgroupCompatibility = 1u << 30, // Subgroup-independent scalar shaders; explicit opt-in.
};
enum BackendFeature : uint32_t {
    FusedWindowAttention = 1u << 0,
    DeviceFeaturePacking = 1u << 1,
    FusedMatrixResidual = 1u << 2,
    PreferUnfusedWindowAttention = 1u << 3, // Scheduling hint, not missing operation support.
    DeviceNoiseCache = 1u << 4,
    FusedQkvPreparation = 1u << 5,
    WorkgroupFp8Matrices = 1u << 6,
    FusedFeedForward = 1u << 7,
    ProjectedQkvPreparation = 1u << 8,
    WorkgroupShapeTiles = 1u << 9,
    WorkgroupCompatibility = 1u << 10
};
}
