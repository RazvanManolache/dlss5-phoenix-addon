#pragma once
#include <algorithm>

namespace phoenix_compat {
// The game's presentation rate is measured before each inference hook. Use
// short windows so a single slow Present does not set the GPU idle budget.
class D3D9InferencePacing {
    static constexpr unsigned window = 24;
    unsigned samples_ = 0;
    unsigned passes_ = 0;
    unsigned gap_us_ = 2000;
    double sum_ms_ = 0;
    double reference_ms_ = 0;

public:
    unsigned gap_us() const { return gap_us_; }
    unsigned frame_budget_us() const { return unsigned(reference_ms_ * 1000); }

    void observe(double interval_ms, unsigned passes, bool started) {
        if (interval_ms <= 0 || interval_ms >= 250) {
            samples_ = 0;
            sum_ms_ = reference_ms_ = 0;
            gap_us_ = 2000;
            return;
        }
        if (passes != passes_) {
            passes_ = passes;
            samples_ = 0;
            sum_ms_ = 0;
            gap_us_ = 2000;
        }
        sum_ms_ += interval_ms;
        if (++samples_ < window) return;
        const double measured = sum_ms_ / window;
        samples_ = 0;
        sum_ms_ = 0;
        if (reference_ms_ == 0 || !started || passes <= 1) {
            reference_ms_ = measured;
            return;
        }

        const double ratio = measured / reference_ms_;
        if (ratio > 1.035)
            gap_us_ = std::min(6500u, gap_us_ + (ratio > 1.08 ? 1000u : 500u));
        else if (ratio < 1.015)
            gap_us_ -= std::min(250u, gap_us_);
        // Refresh on a faster game rate and drift across scene/cap changes.
        reference_ms_ = std::min(measured, reference_ms_ * 1.003);
    }
};
}
