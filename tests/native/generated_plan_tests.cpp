#include <d3d12.h>
#include <wrl.h>
#include "../../src/native/support/network_policy.h"
#include <iostream>
using phoenix_bundle::Json;
static void check(bool condition) {
    if (!condition)
        throw std::runtime_error("Generated plan contract");
}
int main(int argc, char** argv) {
    try {
        check(argc == 2);
        const std::filesystem::path root = argv[1];
        const auto graph =
            phoenix_bundle::json_file(root / "src/native/model/production_native_graph.json");
        const auto contracts =
            phoenix_bundle::json_file(root / "src/native/model/production_kernel_contracts.json");
        Json kernels = Json::object(), weights = {{"records", Json::array()}, {"bytes", 153 * 512}};
        std::vector<phoenix_kernel::Spec> specs;
        for (const auto& kernel : contracts.at("kernels")) {
            const auto name = kernel.at("name").get<std::string>();
            kernels[name] = kernel;
            const auto block = kernel.at("block").get<std::array<unsigned, 3>>();
            specs.push_back({name, "synthetic", "", kernel.at("entry"),
                             kernel.at("parameter_bytes"), block[0], block[1], block[2]});
        }
        std::set<std::string> names;
        for (const auto& layer : graph.at("layers"))
            for (const auto& binding_entry : layer.at("bindings").items()) {
                if (binding_entry.value().at("target").is_null())
                    continue;
                const std::string target = binding_entry.value().at("target");
                if (target.rfind("weight:", 0) == 0)
                    names.insert(target.substr(7));
            }
        check(names.size() == 153);
        // Geometry independently checked against the vendor runner. Include
        // partial frame-boundary lanes and an aligned extent that needs a
        // shifted-window margin; rounding every axis to 128 is not equivalent.
        for (const auto expected : std::vector<std::array<unsigned, 4>>{
                 {320, 320, 320, 320}, {640, 360, 640, 384},
                 {960, 512, 960, 512}, {960, 513, 960, 640},
                 {960, 520, 960, 640}, {960, 521, 960, 576},
                 {960, 528, 960, 576}, {960, 529, 960, 640},
                 {960, 536, 960, 640}, {960, 537, 960, 576},
                 {960, 540, 960, 576},
                 {801, 603, 896, 640}, {1920, 1080, 1920, 1152},
                 {1936, 1072, 1984, 1088}, {1937, 1073, 2048, 1152},
                 {2560, 1440, 2560, 1472}, {3840, 2160, 3840, 2176},
                 {5760, 3240, 5760, 3328}, {7680, 4320, 7744, 4352}}) {
            phoenix_generated::Geometry geometry(expected[0], expected[1]);
            check(geometry.values.at("W0") == expected[2] &&
                  geometry.values.at("H0") == expected[3]);
        }
        unsigned offset = 0;
        for (const auto& name : names) {
            weights["records"].push_back({{"name", name}, {"offset", offset}, {"bytes", 1}});
            offset += 512;
        }
        Json sizes = Json::array();
        for (const auto dimensions : std::vector<std::array<unsigned, 2>>{
                 {640, 384}, {1280, 720}, {1920, 1080}, {2560, 1080},
                 {1, 1}, {127, 129}, {257, 255}, {511, 511}, {801, 603},
                 {1919, 1079}, {2560, 1440}, {3840, 2160}, {5760, 3240},
                 {7680, 4320}, {1080, 1920}, {3440, 1440},
                 {17, 31}, {63, 65}, {319, 321}, {321, 319},
                 {511, 513}, {512, 512}, {513, 511}, {641, 359},
                 {853, 479}, {854, 481}, {959, 539}, {960, 540}, {961, 541},
                 {1023, 767}, {1024, 768}, {1025, 769}, {1365, 767},
                 {1366, 768}, {1367, 769}, {1921, 1081},
                 {1936, 1080}, {1921, 1072}, {721, 1281},
                 {3441, 1439}, {3841, 2161}}) {
            auto frame =
                phoenix_generated::frame(graph, kernels, weights, dimensions[0], dimensions[1]);
            const auto info = phoenix_native::preflight(frame, specs);
            check(info.neural_launches == 154 && info.group_ends.size() == 15);
            const auto extent = frame.at("arenas").at("workspace").at("bytes").get<uint64_t>();
            const auto& allocation = frame.at("allocation");
            const phoenix_generated::Geometry geometry(dimensions[0], dimensions[1]);
            const auto cells = geometry.values.at("T");
            const auto padded_cells = (uint64_t(cells) + 31) / 32 * 32;
            for (const auto& allocation_entry : allocation) {
                const auto& tensor = graph.at("tensors").at(
                    allocation_entry.at("tensor").get<std::string>());
                // Include the repacker's zero-filled token tail, even when
                // its logical spatial extent is only 16-token aligned.
                if (tensor.at("producer").get<unsigned>() >= 57 &&
                    tensor.at("producer").get<unsigned>() <= 97)
                    check(allocation_entry.at("bytes").get<uint64_t>() >=
                          padded_cells * tensor.at("shape").back().get<unsigned>());
                if (tensor.contains("zero_padding") &&
                    (geometry.values.at("H5") > geometry.values.at("H4") / 2 ||
                     geometry.values.at("W5") > geometry.values.at("W4") / 2))
                    check(allocation_entry.at("first_group") == 1 &&
                          allocation_entry.at("last_group") == 154);
                const auto begin = allocation_entry.at("offset").get<uint64_t>(),
                           end = begin + allocation_entry.at("bytes").get<uint64_t>();
                check(begin >= info.counter_bytes && end <= extent && begin % 512 == 0);
                for (const auto& binding_entry : allocation) {
                    if (allocation_entry.at("tensor") == binding_entry.at("tensor") ||
                        allocation_entry.at("last_group").get<unsigned>() <
                            binding_entry.at("first_group").get<unsigned>() ||
                        binding_entry.at("last_group").get<unsigned>() <
                            allocation_entry.at("first_group").get<unsigned>())
                        continue;
                    const auto other = binding_entry.at("offset").get<uint64_t>();
                    check(end <= other ||
                          other + binding_entry.at("bytes").get<uint64_t>() <= begin);
                }
            }
            sizes.push_back({{"width", dimensions[0]},
                             {"height", dimensions[1]},
                             {"workspace_bytes", extent},
                             {"counter_bytes", info.counter_bytes}});
        }
        auto bad = graph;
        bad["layers"][0]["bindings"]["216"]["target"] = "tensor:missing";
        bool rejected = false;
        try {
            phoenix_generated::frame(bad, kernels, weights, 640, 384);
        } catch (const std::exception&) {
            rejected = true;
        }
        check(rejected);
        // Cache hits avoid generation/preflight; resizing cannot grow CPU
        // metadata indefinitely. Failed preflight must not evict a good plan.
        phoenix_generated::PlanCache cache;
        unsigned validations = 0;
        auto validate = [&](const Json& plan) {
            phoenix_native::preflight(plan, specs);
            ++validations;
        };
        auto select = [&](unsigned width, unsigned height) -> const Json& {
            return cache.select(graph, kernels, weights, width, height, validate);
        };
        const auto* original = &select(1920, 1080);
        check(&select(1920, 1080) == original && validations == 1);
        select(801, 603);
        select(2560, 1440);
        select(1080, 1920);
        select(1920, 1080); // Most recently used: retain it when the fifth size arrives.
        select(3840, 2160);
        check(cache.size() == 4 && validations == 5);
        check(&select(1920, 1080) == original && validations == 5);
        rejected = false;
        try {
            cache.select(graph, kernels, weights, 333, 777, [](const Json&) {
                throw std::runtime_error("Rejected plan");
            });
        } catch (const std::exception&) { rejected = true; }
        check(rejected && cache.size() == 4 && &select(1920, 1080) == original);
        select(801, 603); // Previously evicted: regenerate.
        check(validations == 6 && cache.size() == 4);
        for (auto dimensions : {std::array<unsigned, 2>{0, 720}, {1920, 0},
                                {16385, 1080}, {1920, 16385}}) {
            rejected = false;
            try { select(dimensions[0], dimensions[1]); }
            catch (const std::exception&) { rejected = true; }
            check(rejected && cache.size() == 4);
        }
        // Changing an active extent without updating its tile geometry is rejected.
        auto corrupted = phoenix_generated::frame(graph, kernels, weights, 801, 603);
        auto raw = phoenix_bundle::hex(corrupted["launches"][1]["template_hex"]);
        raw[244] ^= 64;
        corrupted["launches"][1]["template_hex"] = phoenix_generated::hex_bytes(raw.data(), raw.size());
        rejected = false;
        try { phoenix_native::preflight(corrupted, specs); }
        catch (const std::exception&) { rejected = true; }
        check(rejected);
        std::cout << Json({{"passed", true},
                           {"cpu_only", true},
                           {"model_files_used", false},
                           {"frames", sizes},
                           {"live_intervals_checked", true},
                           {"padded_token_storage_checked", true},
                           {"persistent_spatial_padding_checked", true},
                           {"bounded_plan_cache_checked", true},
                           {"unknown_tensor_rejected", true}})
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
