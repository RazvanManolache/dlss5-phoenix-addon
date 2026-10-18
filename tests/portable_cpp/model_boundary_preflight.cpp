// Reads model data and initializes the portable CPU backend; never runs a graph.
#include "../../native_runtime/phoenix.h"
#include "../../third_party/nlohmann/json.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using Json = nlohmann::json;
int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("Usage: model_boundary_preflight RUNTIME_ROOT MODEL_DLL SHA256 EXPECTED_GAIN");
        const float expected = std::stof(argv[4]);
        Json configuration = {{"backend", "cpu"}, {"device", "cpu"},
            {"runtime_root", argv[1]}, {"allow_unvalidated", true}, {"cpu_threads", 1},
            {"weight_source", {{"kind", "nvidia_dll"}, {"path", argv[2]}, {"sha256", argv[3]}}}};
        unsigned checks = 0;
        auto check = [&](bool condition, const char* message) {
            ++checks; if (!condition) throw std::runtime_error(message);
        };
        Json result = {{"gpu_used", false}, {"graph_executed", false}, {"model_data_read", true}};
        for (const auto* mode : {"model", "legacy_unit"}) {
            configuration["history_gain_mode"] = mode;
            auto session = std::unique_ptr<void, decltype(&phoenix_runtime_destroy)>(
                phoenix_runtime_create(configuration.dump().c_str(), 2, 1), phoenix_runtime_destroy);
            if (!session) throw std::runtime_error(phoenix_runtime_error());
            auto status = [&] {
                std::vector<char> text(phoenix_runtime_status(session.get(), nullptr, 0));
                phoenix_runtime_status(session.get(), text.data(), text.size());
                return Json::parse(text.data());
            };
            const auto initial = status();
            check(initial.at("model_history_gain").get<float>() == expected, "model tensor gain differs");
            check(initial.at("effective_history_gain").get<float>() ==
                      (std::string(mode) == "model" ? expected : 1.f), "effective gain mode differs");
            float color[6] = {.25f,.25f,.25f,.25f,.25f,.25f}, output[6] = {-9,-9,-9,-9,-9,-9};
            float mask[8] = {0,1,1,1,1,1,1,1};
            phoenix_frame_v2 request{};
            request.struct_bytes = sizeof(request);
            request.version = PHOENIX_FRAME_VERSION_2;
            request.color = {color, sizeof(color), 0, 2, 1, 3, 0};
            request.output = {output, sizeof(output), 0, 2, 1, 3, 0};
            request.metadata_json = "{}";
            auto rejected = [&](const char* reason) {
                check(phoenix_runtime_frame_v2(session.get(), &request) == -1, "invalid frame accepted");
                check(std::string(phoenix_runtime_error()).find(reason) != std::string::npos,
                      "unexpected rejection reason");
                check(status() == initial, "preflight rejection changed runtime state");
                for (float value : output) check(value == -9, "preflight rejection wrote output");
            };
            request.output.capacity_bytes--;
            rejected("capacity");
            request.output.capacity_bytes++;
            request.version = 99;
            rejected("version");
            request.version = PHOENIX_FRAME_VERSION_2;
            request.control_mask = {mask, sizeof(mask), 0, 2, 1, 4, 0};
            rejected("profile");
            request.control_profile = PHOENIX_CONTROL_EXPERIMENTAL_RGB_1;
            mask[3] = 0;
            rejected("alpha");
            mask[3] = 1; mask[0] = NAN;
            rejected("finite");
            mask[0] = 0;
            request.metadata_json = "{\"reset\":true,\"controls\":{\"sampling_mode\":\"invalid\"}}";
            rejected("sampling");
            request.metadata_json = "{\"has_motion\":true}";
            rejected("conflicts");
            result[mode] = initial;
        }
        result["checks"] = checks;
        result["passed"] = true;
        std::cout << result.dump(2) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
