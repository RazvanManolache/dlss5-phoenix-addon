#pragma once
#include "backend.h"
#include "../src/native/support/budget.h"
#include "../third_party/nlohmann/json.hpp"
#include <array>
#include <cmath>

namespace phoenix_cpp {
// Shared scheduling for Metal, Vulkan, CUDA, HIP, SYCL and CPU. Operators and
// learned block numbering remain properties of the selected model graph.
class NetworkSchedule {
    using Json = nlohmann::json;
    phoenix_addon::phoenix::DepthBudget controller_;
    std::string context_;
    bool enabled_ = false, fps_ = false, gpu_backend_ = false;
    double gpu_ms_ = 0;
    bool gpu_sample_ = false;

  public:
    unsigned depth = 8, passes = 1;
    std::array<unsigned, 8> repeats{{1, 1, 1, 1, 1, 1, 1, 1}};
    void prepare(const Json& metadata, const std::string& model_context, bool gpu_backend) {
        gpu_backend_ = gpu_backend;
        demand(metadata.is_object(), "Frame metadata must be an object");
        if (metadata.contains("repeats")) {
            const auto& raw = metadata.at("repeats");
            demand(raw.is_array() && raw.size() == 8,
                   "Expected eight transformer repetition counts");
            for (const auto& count : raw)
                demand(count.is_number_integer() && count >= 0 && count <= 4,
                       "Block repetition outside 0..4");
        }
        for (const auto* field : {"transformer_blocks", "passes"})
            if (metadata.contains(field))
                demand(metadata.at(field).is_number_integer() && metadata.at(field) >= 1 &&
                           metadata.at(field) <= (std::string_view(field) == "passes" ? 10 : 8),
                       "Invalid network depth/pass count");
        const auto counts = metadata.value("repeats", std::array<int, 8>{{1, 1, 1, 1, 1, 1, 1, 1}});
        for (int count : counts)
            demand(count >= 0 && count <= 4, "Block repetition outside 0..4");
        const auto ceiling = metadata.value("transformer_blocks", 8u);
        demand(ceiling >= 1 && ceiling <= 8, "Transformer depth outside 1..8");
        passes = metadata.value("passes", 1u);
        demand(passes >= 1 && passes <= 10, "Network passes outside 1..10");
        const auto budget = metadata.value("adaptive_budget", Json::object());
        demand(budget.is_object(), "Adaptive budget must be an object");
        enabled_ = budget.value("enabled", false);
        const auto mode = budget.value("mode", std::string("gpu_time"));
        demand(mode == "gpu_time" || mode == "presentation_fps", "Unknown adaptive budget mode");
        fps_ = mode == "presentation_fps";
        if (budget.contains("minimum_blocks"))
            demand(budget.at("minimum_blocks").is_number_integer() &&
                       budget.at("minimum_blocks") >= 1 && budget.at("minimum_blocks") <= ceiling,
                   "Invalid adaptive minimum");
        const auto floor = budget.value("minimum_blocks", 1u);
        demand(floor >= 1 && floor <= ceiling, "Adaptive minimum exceeds transformer depth");
        const double target_ms = budget.value("target_ms", 5.0);
        const double target_fps = budget.value("target_fps", 60.0);
        demand(std::isfinite(target_ms) && target_ms >= .1 && target_ms <= 100,
               "GPU-time target outside 0.1..100 ms");
        demand(std::isfinite(target_fps) && target_fps >= 10 && target_fps <= 360,
               "Presentation target outside 10..360 FPS");
        demand(!enabled_ || fps_ || gpu_backend,
               "GPU-time adaptation requires a GPU backend; CPU wall time is not GPU time");
        const auto context = model_context + ":" + std::to_string(passes);
        if (context_ != context) {
            controller_ = {};
            context_ = context;
        }
        controller_.configure(enabled_, ceiling, floor, fps_ ? 1000.0 / target_fps : target_ms,
                              fps_, counts);
        if (enabled_ && fps_) {
            demand(metadata.contains("presentation") && metadata.at("presentation").is_object(),
                   "FPS adaptation requires application presentation timestamps");
            const auto& present = metadata.at("presentation");
            demand(present.contains("timestamp_us") &&
                       present.at("timestamp_us").is_number_unsigned() &&
                       present.contains("active") && present.at("active").is_boolean(),
                   "Presentation requires unsigned timestamp_us and boolean active");
            controller_.present(depth, present.at("timestamp_us").get<uint64_t>(),
                                present.at("active").get<bool>());
        }
        depth = enabled_ ? controller_.depth() : ceiling;
        for (unsigned index = 0; index < repeats.size(); ++index)
            repeats[index] = index < depth ? unsigned(counts[index]) : 0u;
        gpu_sample_ = false;
        gpu_ms_ = 0;
    }
    void complete(const std::vector<Event>& events) {
        // Some queue APIs timestamp CPU devices too. Their event flag does not
        // turn CPU queue duration into a GPU measurement.
        if (!gpu_backend_)
            return;
        for (const auto& event : events) {
            if (!event.gpu)
                continue;
            demand(std::isfinite(event.milliseconds) && event.milliseconds >= 0,
                   "Invalid backend GPU duration");
            gpu_sample_ = true;
            gpu_ms_ += event.milliseconds;
        }
        demand(std::isfinite(gpu_ms_), "Backend GPU duration overflow");
        if (gpu_sample_ && gpu_ms_ > 0)
            controller_.sample(depth, gpu_ms_);
    }
    Json status() const {
        return {{"transformer_blocks", depth},
                {"next_transformer_blocks", controller_.depth()},
                {"adaptive", enabled_},
                {"target_mode", fps_ ? "presentation_fps" : "gpu_time"},
                {"estimate_ms", controller_.ema()},
                {"gpu_sample", gpu_sample_},
                {"gpu_ms", gpu_ms_},
                {"scope", "backend GPU events across all network passes"}};
    }
};
} // namespace phoenix_cpp
