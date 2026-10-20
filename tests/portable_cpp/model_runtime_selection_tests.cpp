// No backend is started: unknown model requests fail before device setup.
#include "../../native_runtime/phoenix.h"
#include "../../src/native/model/weight_source.h"
#include <iostream>

int main() {
    try {
        using phoenix_model::Json;
        const auto bytes = phoenix_runtime_capabilities(nullptr, 0);
        if (!bytes || bytes > 65536) throw std::runtime_error("Invalid capability response");
        std::vector<char> buffer(bytes);
        phoenix_runtime_capabilities(buffer.data(), buffer.size());
        const auto capabilities = Json::parse(buffer.data());
        if (capabilities.at("models") != phoenix_model::model_catalogue())
            throw std::runtime_error("Runtime model catalogue differs from shared registry");
        unsigned checks = 1;
        if (capabilities.at("granular_controls").at("transformer_blocks") != 8 ||
            capabilities.at("granular_controls").at("network_passes") != Json::array({1,10}))
            throw std::runtime_error("Portable runtime omitted granular control discovery");
        ++checks;
        for (const auto* kind : {"files", "nvidia_dll", "model_folder"}) {
            Json source = {{"kind", kind}, {"model", "unknown"}};
            if (source["kind"] == "nvidia_dll")
                source.update({{"path", "not-opened.dll"}, {"sha256", std::string(64, '0')}});
            if (source["kind"] == "model_folder")
                source["path"] = phoenix_model::source_path_string(std::filesystem::absolute("not-opened"));
            const auto configuration = Json({{"weight_source", source}}).dump();
            auto* runtime = phoenix_runtime_create(configuration.c_str(), 1920, 1080);
            if (runtime) {
                phoenix_runtime_destroy(runtime);
                throw std::runtime_error("Unimplemented model started a runtime");
            }
            if (std::string_view(phoenix_runtime_error()).find("Unknown Phoenix DLSS5 model") == std::string_view::npos)
                throw std::runtime_error("Model was not rejected before backend configuration");
            ++checks;
        }
        std::cout << Json({{"passed", true}, {"checks", checks}, {"gpu_used", false}}).dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
