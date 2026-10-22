#pragma once
#include "options.h"
#include <chrono>
#include <limits>
#include <span>
#include <vector>

namespace phoenix_pipeline {
enum class ResetReason { None, FirstFrame, Explicit, SourceChanged, GeometryChanged, Timestamp, Cut };
struct FrameIdentity {
    uint64_t stream = 0, sequence = 0;
    int64_t timestamp_ns = 0;
    unsigned width = 0, height = 0;
};
class Discontinuity {
    bool initialized_ = false;
    FrameIdentity previous_{};
    std::vector<uint8_t> thumbnail_;
    unsigned thumbnail_width_ = 0, thumbnail_height_ = 0;
public:
    void clear() { initialized_ = false; thumbnail_.clear(); }
    // Gray samples are a bounded analysis image, not a full-frame readback.
    // Mean removal rejects uniform exposure flashes; coarse translation search
    // prevents ordinary pans from being mistaken for unrelated images.
    ResetReason observe(FrameIdentity frame, bool explicit_reset, bool detect,
                        std::span<const uint8_t> gray = {}, unsigned w = 0, unsigned h = 0) {
        if (!frame.width || !frame.height || (!gray.empty() &&
            (!w || !h || uint64_t(w) * h != gray.size() || gray.size() > 320u * 180u)))
            throw std::invalid_argument("Invalid discontinuity frame");
        ResetReason reason = !initialized_ ? ResetReason::FirstFrame : ResetReason::None;
        if (initialized_) {
            if (frame.stream != previous_.stream) reason = ResetReason::SourceChanged;
            else if (frame.width != previous_.width || frame.height != previous_.height)
                reason = ResetReason::GeometryChanged;
            else if (frame.sequence <= previous_.sequence || frame.sequence - previous_.sequence > 1 ||
                     frame.timestamp_ns < previous_.timestamp_ns ||
                     uint64_t(frame.timestamp_ns) - uint64_t(previous_.timestamp_ns) > 1000000000ULL)
                reason = ResetReason::Timestamp;
        }
        if (explicit_reset) reason = ResetReason::Explicit;
        if (detect && reason == ResetReason::None && !gray.empty() && w == thumbnail_width_ &&
            h == thumbnail_height_ && thumbnail_.size() == gray.size()) {
            double a = 0, b = 0;
            for (size_t i = 0; i < gray.size(); ++i) { a += gray[i]; b += thumbnail_[i]; }
            const double offset = (a - b) / gray.size();
            double best = std::numeric_limits<double>::infinity();
            for (int dy = -3; dy <= 3; ++dy) for (int dx = -3; dx <= 3; ++dx) {
                double residual = 0; unsigned count = 0;
                for (int y = 3; y + 3 < int(h); y += 2)
                    for (int x = 3; x + 3 < int(w); x += 2) {
                        residual += std::abs(double(gray[size_t(y) * w + x]) -
                            thumbnail_[size_t(y + dy) * w + x + dx] - offset);
                        ++count;
                    }
                if (count) best = std::min(best, residual / (255. * count));
            }
            if (std::isfinite(best) && best > .22) reason = ResetReason::Cut;
        }
        // Turning detection off removes retained pixels and all image work.
        if (detect && !gray.empty()) {
            thumbnail_.assign(gray.begin(), gray.end()); thumbnail_width_ = w; thumbnail_height_ = h;
        } else { thumbnail_.clear(); thumbnail_width_ = thumbnail_height_ = 0; }
        previous_ = frame; initialized_ = true;
        return reason;
    }
};
enum class TimingStage : unsigned { Capture, Motion, Neural, Scale, Present, Count };
struct Telemetry {
    std::array<double, unsigned(TimingStage::Count)> stage_ms{};
    double frame_age_ms = 0;
    uint64_t completed = 0, dropped = 0;
};
class Monitor {
    int mode_ = 0; unsigned interval_ = 30;
    Telemetry data_{};
public:
    void configure(const Options& p) {
        if (!p.valid()) throw std::invalid_argument("Invalid monitoring settings");
        if (mode_ != p.monitoring) data_ = {};
        mode_ = p.monitoring; interval_ = unsigned(p.sample_interval);
    }
    bool enabled() const noexcept { return mode_ != 0; }
    bool gpu_sample(uint64_t frame) const noexcept { return mode_ == 2 && frame % interval_ == 0; }
    void complete(std::span<const double> stages, double age) {
        if (!enabled()) return;
        if (stages.size() != data_.stage_ms.size() || !std::isfinite(age) || age < 0)
            throw std::invalid_argument("Invalid pipeline timing");
        for (double value : stages) if (!std::isfinite(value) || value < 0)
            throw std::invalid_argument("Invalid stage timing");
        std::copy(stages.begin(), stages.end(), data_.stage_ms.begin());
        data_.frame_age_ms = age; ++data_.completed;
    }
    void dropped() noexcept { if (enabled()) ++data_.dropped; }
    const Telemetry& report() const noexcept { return data_; }
};
enum class BudgetKnob { None, Flow, Scale, Blocks, Passes, RestoreFlow, RestoreScale, RestoreBlocks, RestorePasses };
struct BudgetLimits {
    bool flow = false, scale = false, blocks = false, passes = false;
    bool restore_flow = false, restore_scale = false, restore_blocks = false, restore_passes = false;
};
class LatencyPolicy {
    unsigned count_ = 0, below_ = 0, settle_ = 0;
    double age_ = 0;
public:
    void reset() noexcept { count_ = below_ = settle_ = 0; age_ = 0; }
    BudgetKnob sample(const Options& p, double age, bool offline, const BudgetLimits& adjustable) {
        if (offline || p.priority == int(Priority::Fixed) || !p.monitoring || !std::isfinite(age) || age <= 0)
            return BudgetKnob::None;
        if (settle_) { --settle_; return BudgetKnob::None; }
        age_ = (count_ || below_) ? age_ + (age - age_) / 8 : age;
        if (age_ < p.latency_ms * .75) {
            count_=0;
            if (++below_ < 120) return BudgetKnob::None;
            below_=0;
            BudgetKnob restore=BudgetKnob::None;
            if (p.priority == int(Priority::Quality)) {
                if(p.adjust_scale&&adjustable.restore_scale)restore=BudgetKnob::RestoreScale;
                else if(p.adjust_blocks&&adjustable.restore_blocks)restore=BudgetKnob::RestoreBlocks;
                else if(p.adjust_passes&&adjustable.restore_passes)restore=BudgetKnob::RestorePasses;
                else if(p.adjust_flow&&adjustable.restore_flow)restore=BudgetKnob::RestoreFlow;
            } else {
                if(p.adjust_flow&&adjustable.restore_flow)restore=BudgetKnob::RestoreFlow;
                else if(p.adjust_blocks&&adjustable.restore_blocks)restore=BudgetKnob::RestoreBlocks;
                else if(p.adjust_passes&&adjustable.restore_passes)restore=BudgetKnob::RestorePasses;
                else if(p.adjust_scale&&adjustable.restore_scale)restore=BudgetKnob::RestoreScale;
            }
            if(restore!=BudgetKnob::None)settle_=120;
            return restore;
        }
        below_=0;
        if(age_ <= p.latency_ms * 1.1){count_=0;return BudgetKnob::None;}
        if (++count_ < 30) return BudgetKnob::None;
        count_ = 0; settle_ = 60;
        if (p.priority == int(Priority::Quality)) {
            if (p.adjust_flow && adjustable.flow) return BudgetKnob::Flow;
            if (p.adjust_passes && adjustable.passes) return BudgetKnob::Passes;
            if (p.adjust_blocks && adjustable.blocks) return BudgetKnob::Blocks;
            if (p.adjust_scale && adjustable.scale) return BudgetKnob::Scale;
        } else {
            if (p.adjust_passes && adjustable.passes) return BudgetKnob::Passes;
            if (p.adjust_blocks && adjustable.blocks) return BudgetKnob::Blocks;
            if (p.adjust_flow && adjustable.flow) return BudgetKnob::Flow;
            if (p.adjust_scale && adjustable.scale) return BudgetKnob::Scale;
        }
        return BudgetKnob::None;
    }
};
} // namespace phoenix_pipeline
