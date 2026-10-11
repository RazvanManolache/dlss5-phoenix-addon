#include "../../src/native/model/weight_source.h"
#include <iostream>
using namespace phoenix_model;
static void check(bool success) {
    if (!success)
        throw std::runtime_error("DLL loader contract");
}
template <class F> static void rejects(F action) {
    bool failed = false;
    try {
        action();
    } catch (const std::exception&) {
        failed = true;
    }
    check(failed);
}
static void integer(std::vector<uint8_t>& bytes, uint64_t value) {
    const auto byte_offset = bytes.size();
    bytes.resize(byte_offset + 8);
    std::memcpy(bytes.data() + byte_offset, &value, 8);
}
int main() {
    try {
        std::vector<uint8_t> resource(8, 0), expected(153, 0);
        Json catalogue = {{"bytes", 153}, {"unknown_gap_bytes", 0}, {"records", Json::array()}};
        for (unsigned iteration_index = 0; iteration_index < 153; ++iteration_index) {
            const auto name = std::string("synthetic_") + std::to_string(iteration_index);
            const uint8_t value = uint8_t(iteration_index);
            expected[iteration_index] = value;
            integer(resource, name.size());
            resource.insert(resource.end(), name.begin(), name.end());
            integer(resource, 21);
            integer(resource, 21);
            integer(resource, 1);
            resource.insert(resource.end(), 4, 0);
            resource.push_back(value);
            catalogue["records"].push_back({{"name", name},
                                            {"offset", iteration_index},
                                            {"bytes", 1},
                                            {"sha256", weight_digest(&value, 1)}});
        }
        uint64_t size = resource.size();
        std::memcpy(resource.data(), &size, 8);
        catalogue["sha256"] = weight_digest(expected.data(), expected.size());
        check(reconstruct_weights(resource.data(), resource.size(), catalogue) == expected);
        const auto packed = pack_resource(resource.data(), resource.size());
        check(packed.bytes.size() == 153 * 512 && packed.catalogue.at("records").size() == 153);
        std::string previous;
        for (const auto& record : packed.catalogue.at("records")) {
            const auto name = record.at("name").get<std::string>();
            check(previous.empty() || previous < name);
            previous = name;
            const auto number = unsigned(std::stoul(name.substr(10)));
            const auto byte_offset = record.at("offset").get<size_t>();
            check(byte_offset % 512 == 0 && packed.bytes[byte_offset] == uint8_t(number));
            for (size_t pad = 1; pad < 512; ++pad)
                check(packed.bytes[byte_offset + pad] == 0);
        }
        check(WeightSource::config(Json::object()).dll.empty());
        rejects([] { WeightSource::config({{"kind", "unknown"}}); });
        rejects([] {
            WeightSource::config({{"kind", "nvidia_dll"}, {"path", "x.dll"}, {"sha256", ""}});
        });
        rejects([&] { reconstruct_weights(resource.data(), 7, catalogue); });
        rejects([&] { reconstruct_weights(resource.data(), resource.size() - 1, catalogue); });
        auto bad = catalogue;
        bad["sha256"] = std::string(64, '0');
        rejects([&] { reconstruct_weights(resource.data(), resource.size(), bad); });
        auto corrupted = resource;
        corrupted.back() ^= 1;
        rejects([&] { reconstruct_weights(corrupted.data(), corrupted.size(), catalogue); });
        const auto folder = std::filesystem::temp_directory_path() /
                            ("phoenix_model_contract_" + std::to_string(GetCurrentProcessId()) +
                             "_" + std::to_string(GetTickCount64()));
        check(std::filesystem::create_directory(folder));
        struct Cleanup {
            std::filesystem::path folder;
            ~Cleanup() {
                std::error_code e;
                for (const auto* name : {"model.json", "resource.bin"})
                    std::filesystem::remove(folder / name, e);
                std::filesystem::remove(folder, e);
            }
        } cleanup{folder};
        auto source = WeightSource::config({{"kind", "model_folder"}, {"path", folder.u8string()}});
        check(source.external() && source.disk_model() && source.dll.empty());
        check(WeightSource::config(source.configuration()).folder == folder);
        auto read = [](const uint8_t*, size_t) { return 0; };
        auto no_dll = [](const std::filesystem::path&) -> std::string {
            throw std::runtime_error("DLL path unexpectedly used");
        };
        rejects([&] { with_folder_resource(source, read); });
        rejects([] { WeightSource::config({{"kind", "model_folder"}, {"path", "relative"}}); });
        rejects([] { WeightSource::config({{"kind", "model_folder"}, {"path", ""}}); });
        rejects([&] {
            WeightSource::config({{"kind", "model_folder"},
                                  {"path", folder.u8string()},
                                  {"sha256", std::string(64, '0')}});
        });
        auto empty = source;
        empty.folder.clear();
        rejects([&] { empty.validate(); });
        auto mixed = source;
        mixed.dll = L"x.dll";
        rejects([&] { mixed.validate(); });
        Json manifest = {{"schema", "dlss5-phoenix-model/1"},
                         {"architecture", "phoenix-network37/1"},
                         {"resource",
                          {{"path", "resource.bin"},
                           {"bytes", resource.size()},
                           {"sha256", weight_digest(resource.data(), resource.size())}}}};
        auto save = [&](const Json& value) {
            std::ofstream(folder / "model.json", std::ios::binary) << value.dump();
        };
        std::ofstream(folder / "resource.bin", std::ios::binary)
            .write(reinterpret_cast<const char*>(resource.data()), resource.size());
        save(manifest);
        check(folder_manifest(folder) == manifest);
        // Valid checksum but incompatible synthetic names must never reach a decoder.
        rejects([&] { with_model_resource(source, no_dll, read); });
        auto bad_manifest = manifest;
        bad_manifest["resource"]["sha256"] = std::string(64, '0');
        save(bad_manifest);
        rejects([&] { with_folder_resource(source, read); });
        for (const auto& extent : {Json(-1), Json(0), Json(1.5), Json(true), Json("123"),
                                   Json(536870913), Json(resource.size() + 1)}) {
            bad_manifest = manifest;
            bad_manifest["resource"]["bytes"] = extent;
            save(bad_manifest);
            rejects([&] { with_folder_resource(source, read); });
        }
        for (const auto* path : {"../resource.bin", "C:/resource.bin", "resource.bin:stream",
                                 "/resource.bin", "missing.bin"}) {
            bad_manifest = manifest;
            bad_manifest["resource"]["path"] = path;
            save(bad_manifest);
            rejects([&] { with_folder_resource(source, read); });
        }
        bad_manifest = manifest;
        bad_manifest["architecture"] = "unknown";
        save(bad_manifest);
        rejects([&] { folder_manifest(folder); });
        bad_manifest = manifest;
        bad_manifest["schema"] = "unknown";
        save(bad_manifest);
        rejects([&] { folder_manifest(folder); });
        std::ofstream(folder / "model.json", std::ios::binary) << std::string(65537, ' ');
        rejects([&] { folder_manifest(folder); });
        std::cout << Json({{"passed", true},
                           {"cpu_only", true},
                           {"synthetic_records", 153},
                           {"disk_folder_contracts", true},
                           {"model_files_used", false}})
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
