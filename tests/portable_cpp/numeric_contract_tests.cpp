#include <cstdint>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cfenv>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#define PHX_DEVICE
#include "../../native_runtime/host_math.h"
#include "../../native_runtime/fp8_rounding.h"
#include "../../src/backends/hip_sycl/compute_ops.h"

static unsigned fp8_checks = 0, residual_checks = 0;
struct OptimizedMath : HostMath {
    static float round_e4m3(float input) { return phoenix_cpp::round_e4m3_bits<HostMath>(input); }
};
#include "fp8_reference.h"
static void check_fp8(float input) {
    const auto expected = HostMath::bits(reference_fp8(input));
    const phx::Kernel<HostMath> generic{};
    const phx::Kernel<OptimizedMath> optimized{};
    if (HostMath::bits(generic.quantize_e4m3(input)) != expected ||
        HostMath::bits(optimized.quantize_e4m3(input)) != expected)
        throw std::runtime_error("FP8 oracle mismatch at input bits " + std::to_string(HostMath::bits(input)));
    ++fp8_checks;
}
static float decode_half(unsigned bits) {
    const unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    float value = exponent == 31
        ? (fraction ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity())
        : exponent ? std::ldexp(float(1024 + fraction), int(exponent) - 25)
                   : std::ldexp(float(fraction), -24);
    return bits & 32768 ? -value : value;
}
static void residuals() {
    using Kernel = phx::Kernel<HostMath>;
    Kernel math{};
    constexpr float skip = 1.f + 0x1p-10f, gain = skip, projected = -(1.f + 0x1p-9f);
    for (unsigned width : {32u, 64u, 128u, 256u, 512u, 1024u}) {
        // Exercise the actual descriptor dispatch and both publication outputs,
        // rather than only calling the extracted helper under test.
        std::array<float, 1024> coarse{}, fine{}, gains{}, output{};
        std::array<uint16_t, 1024> published{};
        coarse.fill(projected); fine.fill(skip); gains.fill(gain);
        phx::Operation operation{};
        operation.flags = phx::UpsampleResidual;
        operation.column_count = width;
        operation.left_batch_stride = operation.right_batch_stride = 1;
        operation.left_input = coarse.data(); operation.right_input = fine.data();
        operation.auxiliary = gains.data(); operation.output = output.data();
        operation.secondary_output = published.data();
        const float expected = width == 512 ? 0.f : 0x1p-20f;
        Kernel{operation}.unary(0);
        if (HostMath::bits(output[0]) != HostMath::bits(expected) || published[0] != 0)
            throw std::runtime_error("Residual intermediate rounding schedule changed");
        ++residual_checks;
    }
    if (math.upsample_residual(projected, skip, gain, Kernel::ResidualRounding::HalfProductThenAdd) != 0 ||
        math.upsample_residual(projected, skip, gain, Kernel::ResidualRounding::FusedMultiplyAdd) != 0x1p-20f)
        throw std::runtime_error("Explicit residual policies no longer differ");
}
int main() {
    try {
        for (unsigned bits = 0; bits < 65536; ++bits) check_fp8(decode_half(bits));
        for (unsigned code = 0; code < 126; ++code) {
            const auto decode = [](unsigned c) {
                return c / 8 ? std::ldexp(float(8 + c % 8), int(c / 8) - 10)
                             : std::ldexp(float(c), -9);
            };
            const float midpoint = (decode(code) + decode(code + 1)) / 2;
            for (float value : {std::nextafter(midpoint, 0.f), midpoint,
                                std::nextafter(midpoint, std::numeric_limits<float>::infinity())}) {
                check_fp8(value); check_fp8(-value);
            }
        }
        uint32_t random = 0x94f2e175u;
        for (unsigned i = 0; i < 100'000; ++i) {
            random = random * 1664525u + 1013904223u;
            check_fp8(HostMath::value(random));
        }
        for (auto bits : {0x7fc00001u, 0xffc00001u, 0x7f800001u, 0xff800001u,
                          0x7f800000u, 0xff800000u, 0x80000000u, 1u, 0x80000001u})
            check_fp8(HostMath::value(bits));
        const int previous_rounding = std::fegetround();
        for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
            if (std::fesetround(rounding)) throw std::runtime_error("Cannot set rounding mode");
            for (float input : {1.0625f, -1.0625f, 1.1875f, -1.1875f, 0x1p-10f, -0x1p-10f})
                check_fp8(input);
        }
        std::fesetround(previous_rounding);
        residuals();
        std::cout << "{\"passed\":true,\"fp8_oracle_cases\":" << fp8_checks
                  << ",\"residual_dispatch_cases\":" << residual_checks << ",\"gpu_used\":false}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
