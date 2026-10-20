#include "../../src/native/model/model_library.h"
#include "../../src/native/support/routing.h"
#include <iostream>

using namespace phoenix_model;
namespace fs = std::filesystem;
static unsigned checks = 0;
static void expect(bool value) {
    ++checks;
    if (!value) throw std::runtime_error("Shared model library contract failed");
}
static void json_file(const fs::path& path, const Json& value) {
    std::ofstream(path, std::ios::binary) << value.dump(2);
}

int main(int argc, char** argv) {
    const auto root = fs::absolute(fs::temp_directory_path() /
        ("phoenix-model-library-test-" + std::to_string(GetCurrentProcessId())));
    try {
        fs::create_directories(root / "models" / "fixture");
        const auto folder = root / "models" / "fixture";
        const auto manifest = folder / "model.json";
        std::ofstream(folder / "weights.bin", std::ios::binary) << "1234";
        json_file(manifest, {{"schema", "dlss5-phoenix-model/1"}, {"model", "original"},
            {"architecture", models::original.architecture},
            {"resource", {{"path", "weights.bin"}, {"bytes", 4}, {"sha256", std::string(64, 'a')}}}});
        Json entry{{"id", "native-original"}, {"label", "Native Original"}, {"model", "original"},
            {"manifest", "models/fixture/model.json"}, {"manifest_sha256", mapped_file_digest(manifest)}};
        auto publish = [&](Json item) {
            json_file(root / "model-library.json", {{"schema", "phoenix-model-library/1"}, {"models", Json::array({item})}});
        };
        expect(discover_models(root).empty());
        publish(entry); expect(discover_models(root).size() == 1);
        auto broken = entry; broken["manifest"] = "../model.json"; publish(broken);
        expect(discover_models(root).empty());
        broken = entry; broken["model"] = "onnx-fp16"; publish(broken);
        expect(discover_models(root).empty());
        broken = entry; broken["manifest_sha256"] = std::string(64, '0'); publish(broken);
        expect(discover_models(root).empty());
        broken = entry; broken["label"] = std::string("bad\0label", 9); publish(broken);
        expect(discover_models(root).empty());
        publish(entry); fs::remove(folder / "weights.bin"); expect(discover_models(root).empty());
        expect(discover_models(fs::path("relative-library")).empty());
        phoenix_addon::phoenix::Settings settings; settings.model_library_directory = source_path_string(root);
        expect(settings.valid()); bool visited = false;
        phoenix_addon::phoenix::visit_settings(settings, [&](const char* name, auto&) {
            if (std::string_view(name) == "ModelLibraryDirectory") visited = true;
        });
        expect(visited); settings.model_library_directory = std::string(4096, 'x'); expect(!settings.valid());
        unsigned imported = 0;
        if (argc == 3) {
            const auto entries = discover_models(fs::u8path(argv[1])); expect(entries.size() == 2);
            const auto dll = fs::absolute(fs::u8path(argv[2])); const auto dll_sha = mapped_file_digest(dll);
            for (const auto& installed : entries) {
                auto source = WeightSource::config({{"kind", "model_folder"},
                    {"path", source_path_string(installed.folder)}, {"model", installed.model_id}});
                auto actual = packed_model(source, mapped_file_digest);
                WeightSource reference{dll, dll_sha}; reference.model_id = installed.model_id;
                auto expected = dll_packed_model(reference, mapped_file_digest);
                expect(actual.bytes == expected.bytes);
                expect(actual.catalogue.at("records").size() == source.model().record_count);
                ++imported;
            }
        }
        fs::remove_all(root);
        std::cout << Json({{"passed", true}, {"checks", checks}, {"imported_models", imported},
            {"gpu_used", false}, {"games_launched", false}}).dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        fs::remove_all(root);
        return 1;
    }
}
