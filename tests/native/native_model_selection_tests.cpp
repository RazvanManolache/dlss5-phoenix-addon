// Rejection occurs before any D3D12 device creation or model/bundle I/O.
#include "../../native_runtime/phoenix_native.h"
#include "../../third_party/nlohmann/json.hpp"
#include <iostream>

int main() {
    try {
        using Json = nlohmann::json;
        unsigned checks = 0;
        for (const auto* model : {"small-fp8", "unknown"}) {
            const auto config = Json({{"width", 1920}, {"height", 1080},
                {"weight_source", {{"kind", "files"}, {"model", model}}}}).dump();
            auto* session = phoenix_native_create(config.c_str(), nullptr, nullptr);
            if (session) {
                phoenix_native_destroy(session);
                throw std::runtime_error("Unsupported native model started");
            }
            const auto error = std::string(phoenix_native_error());
            const std::string expected = std::string(model) == "small-fp8" ? "requires an external model source" : "Unknown Phoenix DLSS5 model";
            if (error.find(expected) == std::string::npos)
                throw std::runtime_error("Native model rejected too late: " + error);
            ++checks;
        }
        // Preserve legacy original-model file sources. Invalid geometry stops
        // this call before device creation, after source/architecture checks.
        const auto original = Json({{"width", 0}, {"height", 1080},
            {"weight_source", {{"kind", "files"}}}}).dump();
        if (phoenix_native_create(original.c_str(), nullptr, nullptr) ||
            std::string(phoenix_native_error()).find("Invalid native/source extent") == std::string::npos)
            throw std::runtime_error("Original native file-source compatibility changed");
        ++checks;
        std::cout << Json({{"passed", true}, {"checks", checks}, {"gpu_used", false}}).dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
