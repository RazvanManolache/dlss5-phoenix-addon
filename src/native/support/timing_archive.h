#pragma once
#include "../session/session.h"
#include <array>
#include <map>
#include <tuple>

namespace phoenix_addon::phoenix {
// CPU-owned measurements survive GPU recorder/runner replacement. Each sample
// carries the settings used when its commands were recorded, not today's UI.
class TimingArchive {
  public:
    using Key = std::tuple<unsigned, unsigned, unsigned, unsigned, std::string>;
    struct Metric {
        uint64_t samples = 0;
        double total = 0, last = 0, low = 0, high = 0;
        void add(double value) {
            if (!(value > 0) || !std::isfinite(value))
                return;
            if (!samples || value < low)
                low = value;
            high = std::max(high, value);
            last = value;
            total += value;
            ++samples;
        }
        Json json() const {
            return {{"samples", samples},
                    {"mean_ms", samples ? total / double(samples) : 0},
                    {"last_ms", last},
                    {"min_ms", low},
                    {"max_ms", high}};
        }
    };

  private:
    struct Stats {
        Metric route, own, native;
        std::array<Metric, kGroupCount> groups;
        unsigned network_passes = 0;
        uint64_t last_sample = 0;
    };
    struct Recent {
        uint64_t tick = 0;
        Key key{};
        double route = 0, own = 0, native = 0;
    };
    std::map<Key, Stats> rows_;
    std::array<Recent, 120> recent_{};
    std::array<float, 120> history_{};
    unsigned recent_next_ = 0, history_next_ = 0, history_count_ = 0;
    uint64_t epoch_ = 0, sequence_ = 0;

  public:
    uint64_t dropped = 0;
    uint64_t epoch() const {
        return epoch_;
    }
    void clear() {
        ++epoch_; // Already-submitted queries cannot repopulate a cleared table.
        rows_.clear();
        recent_ = {};
        history_count_ = history_next_ = recent_next_ = 0;
        dropped = 0;
    }
    const std::array<float, 120>& history() const {
        return history_;
    }
    unsigned history_count() const {
        return history_count_;
    }
    unsigned history_offset() const {
        return history_count_ < 120 ? 0 : history_next_;
    }
    void sample(const Key& key, uint64_t epoch, uint64_t tick, double route, double own,
                double native, unsigned network_passes,
                const std::array<double, kGroupCount>& groups) {
        if (epoch != epoch_ || !(route > 0) || !std::isfinite(route))
            return;
        // A session can visit many slider combinations. Bound memory without
        // throwing in the render loop; expire the least recently sampled row.
        if (!rows_.count(key) && rows_.size() >= 128) {
            auto oldest = std::min_element(
                rows_.begin(), rows_.end(), [](const auto& left, const auto& right) {
                    return left.second.last_sample < right.second.last_sample;
                });
            rows_.erase(oldest);
        }
        auto& row = rows_[key];
        row.last_sample = ++sequence_;
        row.network_passes = network_passes;
        row.route.add(route);
        row.own.add(own);
        row.native.add(native);
        for (unsigned index = 0; index < kGroupCount; ++index)
            row.groups[index].add(groups[index]);
        history_[history_next_] = float(route);
        history_next_ = (history_next_ + 1) % unsigned(history_.size());
        history_count_ = std::min(unsigned(history_.size()), history_count_ + 1);
        recent_[recent_next_] = {tick, key, route, own, native};
        recent_next_ = (recent_next_ + 1) % unsigned(recent_.size());
    }
    Json rows(uint64_t now) const {
        Json output = Json::array();
        for (const auto& [key, row] : rows_) {
            const auto& [mode, blocks, width, height, context] = key;
            auto settings = Json::parse(context, nullptr, false);
            if (!settings.is_object())
                settings = Json::object();
            Json groups = Json::array();
            for (unsigned index = 0; index < kGroupCount; ++index)
                if (row.groups[index].samples)
                    groups.push_back(
                        {{"group", kGroupLabels[index]}, {"gpu", row.groups[index].json()}});
            Metric recent_route, recent_own, recent_native;
            for (const auto& recent : recent_) {
                if (!recent.tick || recent.key != key || now < recent.tick ||
                    now - recent.tick > 1000)
                    continue;
                recent_route.add(recent.route);
                recent_own.add(recent.own);
                recent_native.add(recent.native);
            }
            output.push_back(
                {{"mode", mode},
                 {"transformer_blocks", blocks},
                 {"width", width},
                 {"height", height},
                 {"network_passes", row.network_passes},
                 {"execution_device", settings.value("ExecutionDevice", std::string("native"))},
                 {"settings", settings},
                 {"route", row.route.json()},
                 {"phoenix", row.own.json()},
                 {"nvidia", row.native.json()},
                 {"groups", groups},
                 {"recent",
                  {{"samples", recent_route.samples},
                   {"route_ms",
                    recent_route.samples ? recent_route.total / double(recent_route.samples) : 0},
                   {"phoenix_samples", recent_own.samples},
                   {"phoenix_ms",
                    recent_own.samples ? recent_own.total / double(recent_own.samples) : 0},
                   {"nvidia_samples", recent_native.samples},
                   {"nvidia_ms", recent_native.samples
                                     ? recent_native.total / double(recent_native.samples)
                                     : 0}}}});
        }
        return output;
    }
};
}
