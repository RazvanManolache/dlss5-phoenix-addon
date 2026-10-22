#include "../../src/native/support/model_selection_state.h"
#include "../../src/native/support/optional_feature_lifetime.h"
#include "../../src/integration/compatibility/host/phoenix-host-config.h"
#include "../../third_party/nlohmann/json.hpp"
#include <iostream>
#include <memory>
#include <vector>

using phoenix_model_tools::ModelSelectionState;
using phoenix_model_tools::replace_idle_generation;
using phoenix_model_tools::retire_optional_feature;

struct Settings {
    std::string model_source = "configured", model_folder, model_dll;
    std::string model_id = "original", nvidia_dll_folder;
};
struct Runner {
    inline static int alive = 0, destroyed = 0;
    bool finished = true;
    std::string model;
    explicit Runner(std::string id) : model(std::move(id)) { ++alive; }
    ~Runner() { --alive; ++destroyed; }
    bool idle() const { return finished; }
};
struct Resource {
    bool present = true;
    void Reset() { present = false; }
};
struct Feature {
    int* handle = nullptr;
    struct Pass { int* handle; Resource output; };
    std::vector<Pass> refinement;
    Resource output, device;
    bool inited = true, failed = true, first = false;
    unsigned attempts = 3, warmup = 8, abandons = 2;
    uint64_t retry_after_tick = 5000;
    std::string requested_contract = "old";
};

int main() try {
    unsigned checks = 0;
    const auto demand = [&](bool value) {
        if (!value) throw std::runtime_error("Model-switch lifetime regression");
        ++checks;
    };
    Settings settings;
    const auto loaded = ModelSelectionState::read(settings);
    // Every source-selector field must invalidate a generation, including the
    // alternate DLL search folder which previously escaped model reloads.
    for (auto field : {&Settings::model_source, &Settings::model_folder,
                       &Settings::model_dll, &Settings::model_id, &Settings::nvidia_dll_folder}) {
        auto changed = settings;
        changed.*field = "changed";
        demand(ModelSelectionState::read(changed) != loaded);
        loaded.restore(changed);
        demand(ModelSelectionState::read(changed) == loaded);
    }
    std::unique_ptr<Runner> active = std::make_unique<Runner>("original");
    auto* previous = active.get();
    unsigned creates = 0, retired = 0;
    auto candidate = [&] { ++creates; return std::make_unique<Runner>("small-fp8"); };
    auto retire = [&](const Runner&) { ++retired; };
    active->finished = false;
    demand(!replace_idle_generation(active, candidate, retire));
    demand(active.get() == previous && creates == 0 && retired == 0 && Runner::alive == 1);
    active->finished = true;
    bool rejected = false;
    try {
        replace_idle_generation(active, []() -> std::unique_ptr<Runner> {
            auto partial = std::make_unique<Runner>("bad");
            throw std::runtime_error("corrupt model / allocation failure");
        }, retire);
    } catch (...) { rejected = true; }
    demand(rejected && active.get() == previous && active->model == "original");
    demand(retired == 0 && Runner::alive == 1);
    rejected = false;
    try { replace_idle_generation(active, [] { return std::unique_ptr<Runner>{}; }, retire); }
    catch (...) { rejected = true; }
    demand(rejected && active.get() == previous && Runner::alive == 1);
    demand(replace_idle_generation(active, candidate, retire));
    demand(active->model == "small-fp8" && Runner::alive == 1 && retired == 1);
    for (unsigned iteration = 0; iteration < 32; ++iteration) {
        const auto model = iteration % 2 ? "original" : "small-fp8";
        demand(replace_idle_generation(active, [&] { return std::make_unique<Runner>(model); }, retire));
        demand(active->model == model && Runner::alive == 1);
    }
    active.reset();
    demand(Runner::alive == 0);

    int handles[3]{};
    Feature feature; feature.handle = &handles[0];
    feature.refinement = {{&handles[1], {}}, {&handles[2], {}}};
    std::vector<int*> releases;
    auto release = [&](int* handle) { releases.push_back(handle); };
    demand(!retire_optional_feature(feature, false, release));
    demand(releases.empty() && feature.output.present && feature.handle == &handles[0]);
    demand(retire_optional_feature(feature, true, release));
    demand(releases.size() == 3 && !feature.handle && feature.refinement.empty());
    demand(!feature.output.present && !feature.device.present && !feature.inited && !feature.failed);
    demand(feature.attempts == 0 && feature.warmup == 0 && feature.retry_after_tick == 0);
    demand(retire_optional_feature(feature, true, release) && releases.size() == 3);
    // A failed release keeps that handle for retry, but must not release an
    // already successful handle twice on the next retirement attempt.
    Feature retry; retry.handle = &handles[0]; retry.refinement = {{&handles[1], {}}};
    bool fail_once = true;
    try {
        retire_optional_feature(retry, true, [&](int* handle) {
            if (handle == &handles[1] && fail_once) { fail_once = false; throw std::runtime_error("busy"); }
            releases.push_back(handle);
        });
    } catch (...) {}
    demand(!retry.handle && retry.refinement[0].handle == &handles[1] && retry.output.present);
    demand(retire_optional_feature(retry, true, release));
    demand(releases.size() == 5 && retry.refinement.empty());
    // Existing released helper binaries use this status token as part of
    // their completion contract. Human-facing qualifications belong in a
    // separate UI field until that protocol is versioned on both sides.
    nlohmann::json before = {{"schema", "production-game-phoenix/2"},
        {"input_provider", "feeder_d3d12"}, {"terminal", false}, {"completed_frames", 10},
        {"phoenix_routing", {{"frames", 10}, {"active_mode", 3}}},
        {"direct_game", {{"enabled", true}, {"status", "Own network output written back to the game DLSS output"}}},
        {"worker_execution", {{"kind", "native"}}}};
    auto after = before;
    after["completed_frames"] = 11;
    after["phoenix_routing"]["frames"] = 11;
    demand(PhoenixOutputAdvanced(before, after));
    demand(!PhoenixOutputAdvanced(before, before));
    after["completed_frames"] = 10;
    demand(!PhoenixOutputAdvanced(before, after));
    after["completed_frames"] = 11;
    after["direct_game"]["status"] = "Waiting for a frame";
    demand(!PhoenixOutputAdvanced(before, after));
    after["direct_game"]["status"] = before["direct_game"]["status"];
    after["terminal"] = true;
    demand(!PhoenixOutputAdvanced(before, after));
    std::cout << nlohmann::json({{"passed", true}, {"checks", checks},
        {"successful_switches", retired}, {"live_executors", Runner::alive},
        {"gpu_used", false}}).dump() << '\n';
    return 0;
} catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n'; return 1;
}
