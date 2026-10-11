#pragma once
#include "policy.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace phoenix_addon::diagnostic {
struct Difference {
    size_t bytes = 0, pixels = 0;
    size_t first_byte = std::numeric_limits<size_t>::max();
    bool exact() const noexcept {
        return bytes == 0;
    }
};
inline std::vector<uint8_t> pack_rows(const uint8_t* data, size_t extent, size_t offset,
                                      size_t pitch, size_t row_bytes, size_t rows) {
    check(data && rows && row_bytes && pitch >= row_bytes && offset <= extent,
          "phoenix diagnostic invalid readback layout");
    check(row_bytes <= extent - offset && rows - 1 <= (extent - offset - row_bytes) / pitch,
          "phoenix diagnostic readback outside allocation");
    check(rows <= std::numeric_limits<size_t>::max() / row_bytes,
          "phoenix diagnostic image extent overflow");
    std::vector<uint8_t> result(rows * row_bytes);
    for (size_t row_index = 0; row_index < rows; ++row_index)
        for (size_t byte_in_row = 0; byte_in_row < row_bytes; ++byte_in_row)
            result[row_index * row_bytes + byte_in_row] =
                data[offset + row_index * pitch + byte_in_row];
    return result;
}
// Bounded HDR error: the byte-exact verdict cannot apply where the
// candidate passes through the HDR bridge (normalize/denormalize fp16
// stores) and a different network than the native DLL. This measures how
// far the candidate is instead: RGB max absolute and max relative error
// over fp16-decoded values, alpha reported separately (its provenance the
// network does not own), and non-finite channels policed structurally --
// a candidate non-finite where the native is finite (invented data) or the
// reverse (clipped data) fails outright, while matched non-finites
// (honest overflow on both sides) are counted, not measured.
//
// The relative floor keeps near-black pixels on the absolute error: with
// paper at 1.0..78x, 0.01 is at most 1% of unit paper. The ceiling is an
// INTERIM sanity bound, not a quality bar: a missing normalize, a missing
// denormalize, or a stuck meter all read ~0.99 relative, so 0.75 catches
// every domain-scale pipeline bug while genuine network-vs-native
// differences get measured on RE9 first. Raws are always saved with a
// bounded_error verdict so that measurement exists; pinning a tighter
// bound from it is the explicit follow-up, not something done here.
constexpr double hdr_rel_floor = 0.01;
constexpr double hdr_sanity_ceiling = 0.75;
struct BoundedError {
    double max_abs = 0.0, max_rel = 0.0, alpha_max_abs = 0.0;
    size_t nonfinite_mismatches = 0, matched_nonfinite = 0;
    bool finite = true;
    bool within(double ceiling) const noexcept {
        return finite && max_rel <= ceiling;
    }
};
inline float decode_half_float(uint16_t bits) noexcept {
    const unsigned mantissa = bits & 0x3ff;
    const unsigned exponent = (bits >> 10) & 0x1f;
    const float sign = (bits & 0x8000) ? -1.0f : 1.0f;
    if (exponent == 0x1f) {
        if (mantissa)
            return std::numeric_limits<float>::quiet_NaN();
        return sign * std::numeric_limits<float>::infinity();
    }
    if (exponent == 0) {
        if (mantissa == 0)
            return sign * 0.0f;
        return sign * std::ldexp(float(mantissa) / 1024.0f, -14);
    }
    return sign * std::ldexp(1.0f + float(mantissa) / 1024.0f, int(exponent) - 15);
}
inline BoundedError bounded_difference(const std::vector<uint8_t>& candidate,
                                       const std::vector<uint8_t>& native) {
    check(!candidate.empty() && candidate.size() == native.size() && candidate.size() % 8 == 0,
          "phoenix diagnostic mismatched RGBA16F extents");
    BoundedError result;
    for (size_t pixel = 0; pixel < candidate.size(); pixel += 8) {
        for (size_t channel = 0; channel < 4; ++channel) {
            uint16_t candidate_bits = uint16_t(candidate[pixel + channel * 2]) |
                                      (uint16_t(candidate[pixel + channel * 2 + 1]) << 8);
            uint16_t native_bits = uint16_t(native[pixel + channel * 2]) |
                                   (uint16_t(native[pixel + channel * 2 + 1]) << 8);
            if (candidate_bits == native_bits)
                continue;
            const float candidate_value = decode_half_float(candidate_bits),
                        native_value = decode_half_float(native_bits);
            const bool c_finite = candidate_value == candidate_value &&
                                  candidate_value != std::numeric_limits<float>::infinity() &&
                                  candidate_value != -std::numeric_limits<float>::infinity();
            const bool n_finite = native_value == native_value &&
                                  native_value != std::numeric_limits<float>::infinity() &&
                                  native_value != -std::numeric_limits<float>::infinity();
            if (!c_finite || !n_finite) {
                if (c_finite != n_finite) {
                    result.finite = false;
                    ++result.nonfinite_mismatches;
                } else
                    ++result.matched_nonfinite;
                continue;
            }
            const double abs_err = std::fabs(double(candidate_value) - double(native_value));
            if (channel < 3) {
                if (abs_err > result.max_abs)
                    result.max_abs = abs_err;
                const double rel = abs_err / (std::fabs(double(native_value)) > hdr_rel_floor
                                                  ? std::fabs(double(native_value))
                                                  : hdr_rel_floor);
                if (rel > result.max_rel)
                    result.max_rel = rel;
            } else if (abs_err > result.alpha_max_abs)
                result.alpha_max_abs = abs_err;
        }
    }
    return result;
}
// Candidate-only statistics: when the native refuses (RE9's driver build
// does, deterministically, on a recorded-into list), there is no A/B to
// compare. These describe the candidate alone over fp16-decoded RGBA:
// finiteness (a broken rescale shows as Inf/NaN, not as a large value),
// value range, and mean magnitude. Pure CPU, pinned below.
struct CandidateStatistics {
    bool finite = true;
    double max_abs = 0.0, mean_abs = 0.0, min_value = 0.0, max_value = 0.0;
    size_t channels = 0;
};
inline CandidateStatistics candidate_statistics(const std::vector<uint8_t>& candidate) {
    check(candidate.size() % 8 == 0, "phoenix diagnostic malformed RGBA16F extent");
    CandidateStatistics result;
    double total = 0.0;
    for (size_t pixel = 0; pixel < candidate.size(); pixel += 8) {
        for (size_t channel = 0; channel < 4; ++channel) {
            const uint16_t bits = uint16_t(candidate[pixel + channel * 2]) |
                                  (uint16_t(candidate[pixel + channel * 2 + 1]) << 8);
            const float decoded_value = decode_half_float(bits);
            const bool fin = decoded_value == decoded_value &&
                             decoded_value != std::numeric_limits<float>::infinity() &&
                             decoded_value != -std::numeric_limits<float>::infinity();
            if (!fin) {
                result.finite = false;
                continue;
            }
            const double sample_value = double(decoded_value);
            if (result.channels == 0 || sample_value < result.min_value)
                result.min_value = sample_value;
            if (result.channels == 0 || sample_value > result.max_value)
                result.max_value = sample_value;
            const double magnitude = std::fabs(sample_value);
            if (magnitude > result.max_abs)
                result.max_abs = magnitude;
            total += magnitude;
            ++result.channels;
        }
    }
    if (result.channels)
        result.mean_abs = total / double(result.channels);
    return result;
}
inline Difference compare_rgba16f(const std::vector<uint8_t>& candidate,
                                  const std::vector<uint8_t>& native) {
    check(!candidate.empty() && candidate.size() == native.size() && candidate.size() % 8 == 0,
          "phoenix diagnostic mismatched RGBA16F extents");
    Difference result;
    for (size_t pixel = 0; pixel < candidate.size(); pixel += 8) {
        bool changed = false;
        for (size_t channel_byte = 0; channel_byte < 8; ++channel_byte) {
            const auto iteration_index = pixel + channel_byte;
            if (candidate[iteration_index] == native[iteration_index])
                continue;
            if (!result.bytes)
                result.first_byte = iteration_index;
            ++result.bytes;
            changed = true;
        }
        result.pixels += changed;
    }
    return result;
}
} // namespace production_game41::diagnostic
