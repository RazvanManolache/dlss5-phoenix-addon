// CPU-only controller, persistence and timestamp-capacity regressions.
#include "../../src/native/support/routing.h"
#include "../../src/native/support/stage_queries.h"
#include <iostream>
#include <stdexcept>

using namespace phoenix_addon::phoenix;
static unsigned checks = 0;
static void verify(bool condition, const char* message) {
    ++checks;
    if (!condition)
        throw std::runtime_error(message);
}
static void samples(DepthBudget& budget, double ms, unsigned count = 30) {
    for (unsigned i = 0; i < count; ++i)
        budget.sample(budget.depth(), ms);
}
int main() {
    try {
        for (const char* model : {"original", "small-fp8"}) {
            const auto profile = model_profile(model);
            const bool small = std::string_view(model) == "small-fp8";
            verify(profile.first_transformer == (small ? 11u : 31u), "Wrong learned block IDs");
            for (unsigned depth = 1; depth <= 8; ++depth)
                for (unsigned block = 0; block < 8; ++block)
                    for (int count = 0; count <= 4; ++count) {
                        Settings settings;
                        settings.model_id = model;
                        settings.experimental = settings.automatic_budget = true;
                        settings.transformer_blocks = depth;
                        settings.minimum_blocks = 1;
                        settings.block_repeats[block] = count;
                        unsigned expected = (small ? 38 : 114) + depth * 5;
                        if (block < depth)
                            expected = unsigned(int(expected) + (count - 1) * 5);
                        verify(settings.valid(), "Model rejected granular settings");
                        verify(profile.scheduled_launches(depth, settings.block_repeats) ==
                                   expected,
                               "Wrong model/depth/repetition launch accounting");
                        Settings restored;
                        visit_settings(settings, [&](const char* name, auto& value) {
                            visit_settings(restored, [&](const char* key, auto& destination) {
                                if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                                             std::decay_t<decltype(destination)>>)
                                    if (std::string_view(name) == key)
                                        destination = value;
                            });
                        });
                        verify(restored.model_id == model &&
                                   restored.block_repeats == settings.block_repeats &&
                                   restored.transformer_blocks == int(depth),
                               "Settings did not round trip");
                    }
            // Both graphs share eight global transformers, but not their fixed cost.
            // A zero tail must not cause a no-op adaptation step.
            std::array<int, 8> repeats{{1, 1, 0, 4, 0, 2, 0, 0}};
            DepthBudget budget;
            budget.configure(true, 8, 2, 5, false, repeats);
            samples(budget, 12);
            verify(budget.depth() == 5, "Budget did not bypass zero-cost tail depth changes");
            samples(budget, 12);
            verify(budget.depth() == 3, "Budget did not remove next active repeated block");
            samples(budget, 12);
            verify(budget.depth() == 3, "Budget crossed its effective minimum work floor");
            samples(budget, 1);
            verify(budget.depth() == 4, "Budget did not recover next active block");
            const auto old_epoch = budget.epoch();
            repeats[3] = 1;
            budget.configure(true, 8, 2, 5, false, repeats);
            verify(budget.depth() == 8 && budget.ema() == 0 && budget.epoch() != old_epoch,
                   "Repeat edit did not reset controller observations");
            for (unsigned i = 0; i < 60; ++i)
                verify(!budget.sample(8, 90, old_epoch), "Old configuration timing accepted");
            verify(budget.depth() == 8 && budget.ema() == 0, "Delayed samples contaminated budget");
            budget
                .reset_samples(); // Model/resolution/processing change with identical depth limits.
            verify(!budget.sample(8, 90, old_epoch), "Reset failed to invalidate pending samples");
            budget.configure(true, 8, 2, 1000.0 / 60, true, repeats);
            uint64_t now = 1000000;
            for (unsigned i = 0; i < 120; ++i) {
                now += 25000;
                budget.present(budget.depth(), now, true);
                budget.sample(budget.depth(), 1);
            }
            verify(budget.depth() == 5, "FPS controller failed to reduce actual repeated work");
            const auto paused_depth = budget.depth();
            for (unsigned i = 0; i < 180; ++i) {
                now += 25000;
                budget.present(budget.depth(), now, false);
            }
            verify(budget.depth() == paused_depth, "Inactive presentation changed model depth");
            repeats.fill(0);
            budget.configure(true, 8, 1, 5, false, repeats);
            samples(budget, 100, 120);
            verify(budget.depth() == 8, "All-skipped transformers caused pointless depth changes");
        }
        StageQueries<15> queries;
        unsigned query = 2; // Route and own begin.
        for (unsigned pass = 0; pass < 10; ++pass)
            for (unsigned group = 0; group < 15; ++group) {
                verify(queries.accepts(group), "Ten-pass group timestamps overflowed");
                queries.append(group, query++);
            }
        verify(query + 4 <= queries.stamps, "Own/native/route endpoints exceed timestamp capacity");
        verify(queries.count == 150 && !queries.accepts(0), "Timestamp limit should be ten passes");
        bool rejected = false;
        try {
            queries.append(0, query);
        } catch (const std::exception&) {
            rejected = true;
        }
        verify(rejected, "Eleventh pass overwrote timestamp storage");
        queries = {};
        verify(!queries.accepts(1), "Out-of-order stage timestamp accepted");
        std::cout << "{\"passed\":true,\"checks\":" << checks
                  << ",\"models\":2,\"gpu_executed\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
