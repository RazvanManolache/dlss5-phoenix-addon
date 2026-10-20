// Read-only picker validation: no runtime extraction, game launch, or GPU execution.
#include "../../src/native/support/model_tools.h"
#include <iostream>
using Json = nlohmann::json;

int main(int argc, char** argv) {
    try {
        Json results = Json::array();
        for (const auto& model : phoenix_model::registry().entries())
            if (!model.reference_dll_sha256.empty() && !phoenix_model::known_model_container(model.reference_dll_sha256))
                throw std::runtime_error("Known model container missing from picker policy");
        if (phoenix_model::known_model_container(std::string(64, '0')))
            throw std::runtime_error("Resource equivalence approved unknown DLL code for picker");
        if (!phoenix_model::known_model_container(phoenix_model::compat_fp16_310_9_16_sha256))
            throw std::runtime_error("Pinned compatibility runtime missing from picker policy");
        for (int i = 1; i < argc; ++i) {
            const auto path = std::filesystem::absolute(std::filesystem::u8path(argv[i]));
            const auto first = phoenix_model_tools::selected_dll_digest(path);
            const auto cached = phoenix_model_tools::selected_dll_digest(path);
            if (first != cached || first != phoenix_embedded::file_digest(path))
                throw std::runtime_error("Picker digest/cache mismatch");
            results.push_back({{"dll_sha256", first}, {"cached_selection_matches", true}});
            if (first == phoenix_model::models::small_fp8.reference_dll_sha256 ||
                first == phoenix_model::compat_fp16_310_9_16_sha256) {
                const auto selected = phoenix_model_tools::selected_dll_digest(path, phoenix_model::models::small_fp8);
                if (selected != first || phoenix_model_tools::selected_dll_digest(path, phoenix_model::models::small_fp8) != first)
                    throw std::runtime_error("Smaller model picker selected the wrong resource/cache entry");
                auto configured = phoenix_model::WeightSource{path, first};
                const auto small_source = phoenix_model_tools::selection("nvidia_dll", "", phoenix_model::source_path_string(path), configured, "small-fp8");
                if (small_source.model_id != "small-fp8") throw std::runtime_error("Picker lost the smaller model identity");
                if (phoenix_model_tools::selected_record_count(small_source) != 77)
                    throw std::runtime_error("Validate model used the original record contract for the smaller model");
                results.back()["small_model_resource_and_cache_verified"] = true;
            }
        }
        const auto folder = std::filesystem::temp_directory_path() /
            ("phoenix-picker-rejection-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(GetTickCount64()));
        if (!std::filesystem::create_directory(folder)) throw std::runtime_error("Picker fixture creation failed");
        struct Cleanup {
            std::filesystem::path folder;
            ~Cleanup() {
                std::error_code error;
                std::filesystem::remove(folder / "invalid.dll", error);
                std::filesystem::remove(folder, error);
            }
        } cleanup{folder};
        { std::ofstream file(folder / "invalid.dll", std::ios::binary); file << "not a DLL"; }
        for (unsigned attempt = 0; attempt < 2; ++attempt) {
            bool rejected = false;
            try { phoenix_model_tools::selected_dll_digest(folder / "invalid.dll"); }
            catch (const std::exception& error) {
                rejected = std::string_view(error.what()).find("Unknown NVIDIA DLSS5 DLL container") != std::string_view::npos;
            }
            if (!rejected) throw std::runtime_error("Picker accepted invalid/cached rejected container");
        }
        if (argc > 1) {
            const auto path = std::filesystem::absolute(std::filesystem::u8path(argv[1]));
            if (phoenix_model_tools::selected_dll_digest(path) != phoenix_embedded::file_digest(path))
                throw std::runtime_error("Picker did not recover after rejected selection");
        }
        std::cout << Json({{"passed", true}, {"cpu_only", true}, {"cached_rejection_and_recovery", true},
                           {"external_models", results}}).dump() << '\n';
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
