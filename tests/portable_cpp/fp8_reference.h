#pragma once
#include <array>
#include <algorithm>
#include <cmath>
// Independent oracle: enumerate the 127 nonnegative finite E4M3 values,
// compare distances, and select the even code at ties. No production masks,
// exponent biases or rounding helper appear in this reference.
static float reference_fp8(float input) {
    if (std::isnan(input)) return 448.f;
    static const auto grid = [] {
        std::array<float, 127> values{};
        for (unsigned code = 0; code < values.size(); ++code) {
            const unsigned exponent = code / 8, fraction = code % 8;
            values[code] = exponent ? std::ldexp(float(8 + fraction), int(exponent) - 10)
                                    : std::ldexp(float(fraction), -9);
        }
        return values;
    }();
    const double magnitude = std::min(double(std::fabs(input)), 448.0);
    auto upper = std::lower_bound(grid.begin(), grid.end(), magnitude);
    unsigned code = unsigned(upper - grid.begin());
    if (code && upper != grid.end()) {
        const double low_distance = magnitude - grid[code - 1];
        const double high_distance = grid[code] - magnitude;
        if (low_distance < high_distance || (low_distance == high_distance && (code & 1))) --code;
    }
    const float value = grid[code];
    return input < 0 ? -value : value;
}
