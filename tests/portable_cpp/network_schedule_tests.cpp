// CPU-only scheduling/timing simulation. Does not load a provider or execute a model.
#include "../../native_runtime/network_schedule.h"
#include "../../src/native/support/worker_budget.h"
#include <iostream>
using namespace phoenix_cpp;
static unsigned checks = 0;
static void check(bool value, const char* message) {
    ++checks;
    demand(value, message);
}
template <class F> static void rejects(F function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected, "Invalid portable schedule accepted");
}
int main() try {
    using Json = nlohmann::json;
    for (const char* backend : {"metal", "vulkan", "hip", "sycl", "cuda"})
        for (const char* model : {"original", "small-fp8"}) {
            const std::string context = std::string(backend) + ":" + model;
            Json metadata{
                {"transformer_blocks", 8},
                {"repeats", {1, 1, 0, 4, 0, 2, 0, 0}},
                {"passes", 10},
                {"adaptive_budget", {{"enabled", true}, {"target_ms", 5}, {"minimum_blocks", 2}}}};
            NetworkSchedule schedule;
            for (unsigned frame = 0; frame < 31; ++frame) {
                schedule.prepare(metadata, context, true);
                schedule.complete({{UINT32_MAX, 0, 4, true}, {11, 0, 8, true}, {0, 0, 999, false}});
            }
            check(schedule.depth == 5 && schedule.repeats[5] == 0 && schedule.repeats[3] == 4,
                  "Portable GPU budget lost repeat counts or failed to trim disabled tail");
            check(schedule.passes == 10 && schedule.status().at("gpu_ms") == 12,
                  "Portable budget must sum GPU events across passes, excluding CPU events");
            schedule.prepare(metadata, context + ":new-model-resource", true);
            check(schedule.depth == 8 && schedule.status().at("estimate_ms") == 0,
                  "Model/resource change reused budget measurements");
            for (unsigned frame = 0; frame < 60; ++frame) {
                schedule.prepare(metadata, context, true);
                schedule.complete({{0, 0, 99, false}});
            }
            check(schedule.depth == 8 && !schedule.status().at("gpu_sample").get<bool>(),
                  "CPU wall duration substituted for absent GPU timestamps");
            rejects([&] { schedule.prepare(metadata, context, false); });
            metadata["adaptive_budget"]["mode"] = "presentation_fps";
            rejects([&] { schedule.prepare(metadata, context, true); });
            for (uint64_t frame = 0; frame < 120; ++frame) {
                metadata["presentation"] = {{"timestamp_us", uint64_t(1000000 + frame * 25000)},
                                            {"active", true}};
                schedule.prepare(metadata, context, true);
                schedule.complete({{0, 0, .01, true}});
            }
            check(schedule.depth == 5, "Caller presentation FPS did not control portable depth");
            metadata["presentation"]["active"] = false;
            for (uint64_t frame = 0; frame < 120; ++frame) {
                metadata["presentation"]["timestamp_us"] = uint64_t(5000000 + frame * 25000);
                schedule.prepare(metadata, context, true);
            }
            check(schedule.depth == 5, "Inactive caller presentations changed portable depth");
            metadata["repeats"][3] = -1;
            rejects([&] { schedule.prepare(metadata, context, true); });
        }
    for (unsigned depth = 1; depth <= 8; ++depth) {
        NetworkSchedule schedule;
        schedule.prepare({{"transformer_blocks", depth}, {"repeats", {0, 4, 1, 2, 0, 3, 1, 4}}},
                         "fixed", false);
        check(schedule.depth == depth, "Fixed CPU/portable depth unsupported");
        for (unsigned index = depth; index < 8; ++index)
            check(schedule.repeats[index] == 0, "Truncated portable block retained");
    }
    {
        NetworkSchedule cpu;
        cpu.prepare(Json::object(), "sycl:cpu", false);
        cpu.complete({{0, 0, 500, true}});
        check(!cpu.status().at("gpu_sample").get<bool>() && cpu.status().at("gpu_ms") == 0,
              "SYCL CPU queue events mislabeled as GPU timing");
        for (const auto& invalid :
             {Json{{"repeats", {0, 1, 2}}}, Json{{"repeats", {1, 1, 1, 1, 1, 1, 1, 1.5}}},
              Json{{"transformer_blocks", uint64_t(4294967304)}}, Json{{"passes", 1.5}}})
            rejects([&] { cpu.prepare(invalid, "invalid", true); });
    }
    phoenix_addon::phoenix::DepthBudget budget;
    budget.configure(true, 8, 1, 5);
    phoenix_addon::phoenix::WorkerBudgetFeedback feedback;
    Json execution{{"worker_pid", 42},
                   {"generation", 3},
                   {"last_discard_reason", ""},
                   {"last_result",
                    {{"ok", true},
                     {"frame_id", 1},
                     {"generation", 3},
                     {"model_id", "small-fp8"},
                     {"budget_epoch", budget.epoch()},
                     {"requested_transformer_blocks", 8},
                     {"network_schedule", {{"gpu_sample", true}, {"gpu_ms", 12}}}}}};
    for (unsigned i = 0; i < 60; ++i)
        feedback.sample(budget, execution, "small-fp8");
    check(budget.depth() == 8, "Held worker output counted as multiple GPU samples");
    for (unsigned i = 2; i <= 30; ++i) {
        execution["last_result"]["frame_id"] = i;
        feedback.sample(budget, execution, "small-fp8");
    }
    check(budget.depth() == 7, "Completed portable worker GPU samples did not adapt host");
    budget.reset_samples();
    for (unsigned i = 31; i <= 70; ++i) {
        execution["last_result"]["frame_id"] = i;
        feedback.sample(budget, execution, "small-fp8");
    }
    check(budget.depth() == 7 && budget.ema() == 0,
          "Old worker epoch contaminated new configuration");
    for (unsigned failure = 0; failure < 4; ++failure) {
        auto invalid = execution;
        invalid["last_result"]["budget_epoch"] = budget.epoch();
        invalid["last_result"]["requested_transformer_blocks"] = 7;
        if (failure == 0)
            invalid["generation"] = 4;
        if (failure == 1)
            invalid["last_discard_reason"] = "invalidated_history";
        if (failure == 2)
            invalid["last_result"]["model_id"] = "original";
        if (failure == 3)
            invalid["last_result"]["network_schedule"]["gpu_sample"] = false;
        for (unsigned i = 100; i < 140; ++i) {
            invalid["last_result"]["frame_id"] = i;
            feedback.sample(budget, invalid, "small-fp8");
        }
        check(budget.ema() == 0, "Invalid worker result accepted for budget");
    }
    execution["last_discard_reason"] = "result_age_limit";
    execution["last_result"]["budget_epoch"] = budget.epoch();
    execution["last_result"]["requested_transformer_blocks"] = 7;
    for (unsigned i = 200; i < 230; ++i) {
        execution["last_result"]["frame_id"] = i;
        feedback.sample(budget, execution, "small-fp8");
    }
    check(budget.depth() == 6, "Completed GPU cost of an expired image could not reduce workload");
    std::cout << "{\"passed\":true,\"checks\":" << checks << ",\"gpu_executed\":false}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
