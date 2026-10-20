#pragma once
#include "../src/backends/hip_sycl/fp8_grid.h"
namespace phoenix_cpp {
template <class FloatBits> PHX_DEVICE static float round_e4m3_bits(float input) {
    return phx::round_e4m3_grid<FloatBits>(input);
}
}
