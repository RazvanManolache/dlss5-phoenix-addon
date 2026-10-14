// Real candidate integrity/CPU inference validation without a graphics helper.
// Both bitnesses load model-free embedded runtimes; the model stays external.
#define NOMINMAX
#include "../../src/native/support/model_tools.h"
#include <iostream>
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Supply weight_source.json");
        nlohmann::json config;
        std::ifstream(argv[1]) >> config;
        const auto source = phoenix_model::WeightSource::config(
            config.contains("weight_source") ? config["weight_source"] : config);
        phoenix_model_tools::Task task;
        auto wait = [&] {
            const auto deadline = GetTickCount64() + 180000;
            auto result = task.poll();
            while (result.busy && GetTickCount64() < deadline) {
                Sleep(20);
                result = task.poll();
            }
            if (result.busy) throw std::runtime_error("Candidate validation timed out");
            return result;
        };
        task.validate(source, {}, GetModuleHandleW(nullptr));
        const auto valid = wait();
        if (!valid.passed) throw std::runtime_error(valid.message);
        // A separate invalid candidate must not retain the previous success.
        const auto missing = std::filesystem::temp_directory_path() /
            (L"MissingPhoenixCandidate-" + std::to_wstring(GetCurrentProcessId()));
        const auto invalid = phoenix_model::WeightSource::config(
            {{"kind", "model_folder"}, {"path", phoenix_model::source_path_string(missing)}});
        task.validate(invalid, {}, GetModuleHandleW(nullptr));
        const auto rejected = wait();
        if (rejected.passed || rejected.key == valid.key || rejected.message.empty())
            throw std::runtime_error("Invalid candidate inherited another model's success");
        std::cout << nlohmann::json{{"passed", true}, {"bits", sizeof(void*) * 8},
            {"graphics_helper_required", false}, {"gpu_inference_executed", false},
            {"valid_candidate", valid.message}, {"invalid_candidate_rejected", true}}.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
