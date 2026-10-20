// CPU-only exhaustive tests of the pre-Ada packed conversion path. No CUDA API
// calls, device allocation, kernel launch or model data are used.
#include "../../src/native/kernels/production/common/architecture_math.cuh"
#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

static float decode_code(unsigned code) {
    const unsigned magnitude = code & 127;
    if (magnitude == 127) return NAN;
    const unsigned exponent = magnitude / 8, fraction = magnitude % 8;
    const float value = exponent ? std::ldexp(float(8 + fraction), int(exponent) - 10)
                                 : std::ldexp(float(fraction), -9);
    return code & 128 ? -value : value;
}
static unsigned reference_code(uint16_t half) {
    const __half_raw raw{half};
    const float value = __half2float(__half(raw));
    if (std::isnan(value)) return 127;
    static const auto grid = [] {
        std::array<float, 127> values{};
        for (unsigned i = 0; i < 127; ++i) values[i] = decode_code(i);
        return values;
    }();
    const double magnitude = std::min(double(std::fabs(value)), 448.0);
    auto upper = std::lower_bound(grid.begin(), grid.end(), magnitude);
    unsigned code = unsigned(upper - grid.begin());
    if (code && upper != grid.end()) {
        const double low_distance = magnitude - grid[code - 1];
        const double high_distance = grid[code] - magnitude;
        if (low_distance < high_distance || (low_distance == high_distance && (code & 1))) --code;
    }
    return code | ((half & 0x8000) ? 128 : 0);
}
int main() {
    try {
        for (unsigned low = 0; low < 65536; ++low) {
            const unsigned high = low ^ 65535;
            const unsigned actual = phoenix_architecture::encode_fp8_pair(low | (high << 16));
            for (unsigned lane = 0; lane < 2; ++lane) {
                const unsigned expected = reference_code(uint16_t(lane ? high : low));
                const unsigned code = (actual >> (lane * 8)) & 255;
                if ((expected == 127 ? (code & 127) : code) != expected)
                    throw std::runtime_error("Compatibility E4M3 encode/packing differs from independent grid oracle");
            }
        }
        for (unsigned low = 0; low < 256; ++low) {
            const unsigned high = low ^ 255;
            const auto actual = phoenix_architecture::decode_fp8_pair(uint16_t(low | (high << 8)));
            for (unsigned lane = 0; lane < 2; ++lane) {
                const __half_raw raw{uint16_t(actual >> (lane * 16))};
                const float value = __half2float(__half(raw));
                const float expected = decode_code(lane ? high : low);
                if (std::isnan(expected) ? !std::isnan(value) :
                    value != expected || std::signbit(value) != std::signbit(expected))
                    throw std::runtime_error("Compatibility E4M3 decode/packing differs from grid oracle");
            }
        }
        std::cout << "{\"passed\":true,\"cpu_only\":true,\"half_patterns\":65536,\"fp8_codes\":256}\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
