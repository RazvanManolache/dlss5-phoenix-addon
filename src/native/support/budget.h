#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace phoenix_addon::phoenix {
// Completed GPU samples or active presentation intervals. A depth change never skips an Evaluate, and a
// configuration/mode change starts a fresh observation window.
class DepthBudget {
    unsigned depth_ = 8, ceiling_ = 8, floor_ = 1, count_ = 0;
    double target_ = 0, ema_ = 0;
    bool enabled_ = false;
    bool fps_mode_ = false;
    std::array<int, 8> repeats_{{1, 1, 1, 1, 1, 1, 1, 1}};
    uint64_t epoch_ = 0;
    uint64_t present_tick_ = 0;
    double present_elapsed_ = 0;
    unsigned present_count_ = 0, slow_windows_ = 0, good_windows_ = 0, recovery_wait_ = 0;
    unsigned lower_depth() const noexcept {
        // Removing a skipped tail block saves no work. Find the next boundary
        // that actually removes a repetition, without crossing the user's floor.
        for (unsigned next = depth_; next > floor_;) {
            --next;
            if (repeats_[next] > 0)
                return next;
        }
        return depth_;
    }
    unsigned higher_depth() const noexcept {
        for (unsigned next = depth_ + 1; next <= ceiling_; ++next)
            if (repeats_[next - 1] > 0)
                return next;
        return depth_;
    }
    void reset_present() noexcept {
        present_tick_ = 0;
        present_elapsed_ = 0;
        present_count_ = 0;
        slow_windows_ = good_windows_ = 0;
    }

  public:
    void configure(bool enabled, unsigned ceiling, unsigned floor, double target,
                   bool fps_mode = false,
                   const std::array<int, 8>& repeats = {{1, 1, 1, 1, 1, 1, 1, 1}}) {
        if (enabled_ == enabled && ceiling_ == ceiling && floor_ == floor && target_ == target &&
            fps_mode_ == fps_mode && repeats_ == repeats)
            return;
        enabled_ = enabled;
        ceiling_ = ceiling;
        floor_ = floor;
        target_ = target;
        fps_mode_ = fps_mode;
        repeats_ = repeats;
        depth_ = ceiling;
        reset_samples();
    }
    void reset_samples() noexcept {
        ++epoch_;
        count_ = 0;
        ema_ = 0;
        recovery_wait_ = 0;
        reset_present();
    }
    uint64_t epoch() const noexcept {
        return epoch_;
    }
    bool sample(unsigned depth, double milliseconds, uint64_t epoch) noexcept {
        return epoch == epoch_ && sample(depth, milliseconds);
    }
    bool sample(unsigned depth, double milliseconds) noexcept {
        if (!enabled_ || fps_mode_ || depth != depth_ || !(milliseconds > 0) ||
            !std::isfinite(milliseconds))
            return false;
        ema_ = count_ ? ema_ + (milliseconds - ema_) / 8.0 : milliseconds;
        if (++count_ < 30)
            return false;
        count_ = 0;
        if (ema_ > target_ * 1.10 && lower_depth() != depth_) {
            depth_ = lower_depth();
            ema_ = 0;
            return true;
        }
        if (ema_ < target_ * .75 && higher_depth() != depth_) {
            depth_ = higher_depth();
            ema_ = 0;
            return true;
        }
        return false;
    }
    unsigned depth() const noexcept {
        return depth_;
    }
    // Whole-game presentation intervals, never 1000 / neural-network GPU ms.
    // Ignore inactive/stale routes and pauses, and never mix depth windows.
    bool present(unsigned depth, uint64_t now_us, bool active) noexcept {
        if (!enabled_ || !fps_mode_)
            return false;
        if (!active || depth != depth_ || !present_tick_ || now_us <= present_tick_ ||
            now_us - present_tick_ > 250000) {
            reset_present();
            ema_ = 0;
            if (active && depth == depth_)
                present_tick_ = now_us;
            return false;
        }
        present_elapsed_ += double(now_us - present_tick_) / 1000.0;
        present_tick_ = now_us;
        ++present_count_;
        if (present_elapsed_ < 1000 || present_count_ < 10)
            return false;
        ema_ = present_elapsed_ / double(present_count_);
        present_elapsed_ = 0;
        present_count_ = 0;
        if (recovery_wait_)
            --recovery_wait_;
        if (ema_ > target_ * 1.05) {
            good_windows_ = 0;
            if (++slow_windows_ >= 2 && lower_depth() != depth_) {
                depth_ = lower_depth();
                recovery_wait_ = 5;
                reset_present();
                return true;
            }
            slow_windows_ = std::min(slow_windows_, 2u);
        } else {
            slow_windows_ = 0;
            good_windows_ = ema_ <= target_ * 1.02 ? good_windows_ + 1 : 0;
            // Probe recovery even at a frame cap; spare FPS is hidden there.
            const unsigned required = ema_ < target_ * .90 ? 2u : 5u;
            if (!recovery_wait_ && good_windows_ >= required && higher_depth() != depth_) {
                depth_ = higher_depth();
                reset_present();
                return true;
            }
            good_windows_ = std::min(good_windows_, 5u);
        }
        return false;
    }
    double ema() const noexcept {
        return ema_;
    }
};
}
