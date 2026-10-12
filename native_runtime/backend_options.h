#pragma once
#include <cstdint>
namespace phoenix_cpp {
// Optional factory flags leave the original backend ABI usable by older packages.
enum BackendOption : uint32_t {
    ScalarMatrices = 1u << 0,
    DispatchProfiling = 1u << 1,
    OrderedAttention = 1u << 2,
};
enum BackendFeature : uint32_t {
    FusedWindowAttention = 1u << 0,
    DeviceFeaturePacking = 1u << 1,
    FusedMatrixResidual = 1u << 2
};
}
