#pragma once
#include "budget.h"
#include "../../../third_party/nlohmann/json.hpp"
#include <string>
#include <string_view>

namespace phoenix_addon::phoenix {
// Consume a completed inference once. A held/reprojected result and an old
// generation/configuration must never provide extra controller observations.
class WorkerBudgetFeedback {
    std::string last_;

  public:
    bool sample(DepthBudget& budget, const nlohmann::json& execution, std::string_view model) {
        const auto result = execution.value("last_result", nlohmann::json::object());
        const auto discarded = execution.value("last_discard_reason", std::string());
        // An expired image is still a completed cost sample for this exact
        // configuration. Using its duration can help an initially slow model
        // reach the delivery window; this does not authorize displaying it.
        if (!result.value("ok", false) || (!discarded.empty() && discarded != "result_age_limit") ||
            !result.contains("frame_id") || !result.contains("generation"))
            return false;
        const auto identity = std::to_string(execution.value("worker_pid", uint64_t(0))) + ":" +
                              std::to_string(result.at("generation").get<uint64_t>()) + ":" +
                              std::to_string(result.at("frame_id").get<uint64_t>());
        if (identity == last_)
            return false;
        last_ = identity;
        if (result.at("generation") != execution.value("generation", uint64_t(0)) ||
            result.value("model_id", std::string()) != model || !result.contains("budget_epoch") ||
            !result.contains("network_schedule"))
            return false;
        const auto& timing = result.at("network_schedule");
        if (!timing.value("gpu_sample", false))
            return false;
        return budget.sample(result.value("requested_transformer_blocks", 8u),
                             timing.value("gpu_ms", 0.0),
                             result.at("budget_epoch").get<uint64_t>());
    }
};
} // namespace phoenix_addon::phoenix
