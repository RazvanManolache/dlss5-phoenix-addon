#pragma once
#include "../session/session.h"
#include <map>
#include <string>

namespace phoenix_addon::phoenix {
// Worker wall time is independent of render-queue timestamp intervals.
// Polling the same completed result never contributes another timing sample.
class WorkerTiming {
    struct Metric {
        uint64_t samples = 0;
        double total = 0;
        void add(double duration_ms) {
            if (std::isfinite(duration_ms) && duration_ms >= 0) {
                ++samples;
                total += duration_ms;
            }
        }
        Json json() const {
            return {{"samples", samples}, {"mean_ms", samples ? total / double(samples) : 0}};
        }
    };
    struct Row {
        std::string label, state, error, identity, actual;
        Metric elapsed, prepare, execute, resolve, age;
        uint64_t completions = 0, failed = 0;
    };
    std::map<std::string, Row> rows_;

  public:
    void clear() {
        rows_.clear();
    }
    void sample(const std::string& key, const std::string& label, const Json& execution) {
        if (!rows_.count(key) && rows_.size() >= 64)
            return;
        auto& row = rows_[key];
        row.label = label;
        row.state = execution.value("state", std::string());
        row.error = execution.value("error", std::string());
        const auto result = execution.value("last_result", Json::object());
        if (!result.contains("frame_id"))
            return;
        row.actual = result.value("kernel_engine", std::string());
        const auto identity = std::to_string(execution.value("worker_pid", uint64_t(0))) + ":" +
                              std::to_string(result.value("generation", uint64_t(0))) + ":" +
                              std::to_string(result.at("frame_id").get<uint64_t>());
        if (!result.value("ok", false))
            row.error = result.value("error", std::string());
        if (row.identity == identity)
            return;
        row.identity = identity;
        if (!result.value("ok", false)) {
            ++row.failed;
            row.error = result.value("error", std::string());
            return;
        }
        ++row.completions;
        if (result.contains("elapsed_ms"))
            row.elapsed.add(result.at("elapsed_ms").get<double>());
        if (execution.contains("last_result_age_ms"))
            row.age.add(execution.at("last_result_age_ms").get<double>());
        if (result.contains("phase_ms")) {
            const auto& phase_times = result.at("phase_ms");
            if (phase_times.contains("features_ms"))
                row.prepare.add(phase_times.at("features_ms").get<double>());
            if (phase_times.contains("graph_roundtrip_ms"))
                row.execute.add(phase_times.at("graph_roundtrip_ms").get<double>());
            if (phase_times.contains("resolve_ms"))
                row.resolve.add(phase_times.at("resolve_ms").get<double>());
        }
    }
    Json report() const {
        Json rows = Json::array();
        for (const auto& [key, row] : rows_)
            rows.push_back({{"key", key},
                            {"label", row.label},
                            {"state", row.state},
                            {"error", row.error},
                            {"actual_engine", row.actual},
                            {"completed", row.completions},
                            {"failed", row.failed},
                            {"worker", row.elapsed.json()},
                            {"prepare", row.prepare.json()},
                            {"execute", row.execute.json()},
                            {"resolve", row.resolve.json()},
                            {"age", row.age.json()}});
        return {
            {"scope",
             "Worker wall time and capture-to-result age; not render-queue GPU time or whole-game frame time"},
            {"rows", rows}};
    }
};
}
