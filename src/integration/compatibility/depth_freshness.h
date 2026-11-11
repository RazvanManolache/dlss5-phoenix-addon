#pragma once
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

namespace phoenix_compat {
// Hold the color signature from when these depth samples last changed.
// Repeated depth is normal for a still view. Reject it only when the color
// structure changed broadly, even if the change accumulated over many frames.
class DepthFreshness {
    static constexpr unsigned columns = 24, rows = 14;
    using Signature = std::array<int, columns * rows>;
    Signature reference_{};
    uint64_t depth_signature_ = 0;
    bool initialized_ = false;

  public:
    void reset() { initialized_ = false; }
    bool accept(std::span<const uint8_t> rgba, std::span<const float> depth, unsigned width, unsigned height) {
        const std::size_t pixels = std::size_t(width) * height;
        if (!pixels || depth.size() != pixels || rgba.size() != pixels * 4) return false;
        Signature color{};
        uint64_t signature = 14695981039346656037ull;
        for (unsigned y = 0; y < rows; ++y) {
            for (unsigned x = 0; x < columns; ++x) {
                const std::size_t pixel = std::size_t((2 * y + 1) * height / (2 * rows)) * width +
                    (2 * x + 1) * width / (2 * columns);
                const auto* rgb = rgba.data() + pixel * 4;
                color[y * columns + x] = (77 * rgb[0] + 150 * rgb[1] + 29 * rgb[2]) >> 8;
                signature = (signature ^ std::bit_cast<uint32_t>(depth[pixel])) * 1099511628211ull;
            }
        }
        if (!initialized_ || signature != depth_signature_) {
            initialized_ = true;
            depth_signature_ = signature;
            reference_ = color;
            return true;
        }
        unsigned changed = 0, total = 0;
        for (unsigned y = 1; y < rows; ++y) {
            for (unsigned x = 1; x < columns; ++x) {
                const unsigned i = y * columns + x;
                const int dx = (color[i] - color[i - 1]) - (reference_[i] - reference_[i - 1]);
                const int dy = (color[i] - color[i - columns]) - (reference_[i] - reference_[i - columns]);
                const unsigned difference = unsigned(std::abs(dx) + std::abs(dy));
                changed += difference >= 28;
                total += difference;
            }
        }
        constexpr unsigned comparisons = (columns - 1) * (rows - 1);
        return changed * 4 < comparisons || total < 24 * comparisons;
    }
};
} // namespace phoenix_compat
