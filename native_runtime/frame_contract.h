#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace phoenix_cpp {
inline constexpr const char* frame_boundary_contract = "phoenix-portable/3";

// Right/bottom padding reflects once and then clamps. This is deliberately not
// periodic mirror addressing; far padding and extent-one images must stay put.
inline int reflect_once_then_clamp(int coordinate, int extent) {
    return std::clamp(coordinate < extent ? coordinate : 2 * extent - coordinate - 2,
                      0, extent - 1);
}

// Each half conversion is observable. Combining this into one final conversion
// changes halfway cases and therefore the network's conditioning input.
template <class Math>
inline float centered_half_feature(Math& math, float value) {
    const float rounded_input = math.half(value);
    const float centered = math.half(rounded_input - .5f);
    return math.half(centered * .125f);
}
// Shared by execution and diagnostic stage capture. Invalid history never reads
// or multiplies a head value (including NaN on an unused gate).
inline float resolve_gate(float logit, bool valid, float model_gain) {
    return valid ? (1.f / (1.f + std::exp(-logit))) * model_gain : 0.f;
}

inline float checked_model_history_gain(float value) {
    if (!std::isfinite(value) || value < 0 || value > 1)
        throw std::runtime_error("Model blend_scale must be a finite scalar in 0..1");
    return value;
}

// Candidate indices are local to an evaluation. Only successful publication
// commits the returned next index; fixed replay deliberately reuses one seed.
struct SamplingPlan {
    uint32_t base, first;
    bool advancing;
    uint32_t seed(unsigned pass) const {
        return advancing ? base + first + uint32_t(pass) : base;
    }
    uint32_t next(unsigned passes) const {
        return advancing ? first + uint32_t(passes) : first;
    }
};
}
