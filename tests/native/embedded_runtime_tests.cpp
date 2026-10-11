// Verify and install the embedded model-free payload into an explicit test cache.
#include "../../src/native/support/embedded_runtime.h"
#include <iostream>
int wmain(int argc, wchar_t **argv) {
    using namespace phoenix_embedded;
    try {
        require(argc == 4, "Usage: ADDON CACHE REPORT");
        HMODULE module = LoadLibraryExW(argv[1], nullptr,
                                        LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE);
        require(module != nullptr, "Addon resource image did not load");
        const auto [data, bytes] = resource(module, 1001);
        require(data && bytes, "Addon has no embedded runtime manifest");
        auto manifest = Json::parse(data, data + bytes);
        require(!manifest.at("model_data_included").get<bool>() &&
                    !manifest.at("nvidia_dll_included").get<bool>(),
                "Model payload unexpectedly present");
        auto installed = install(module, fs::absolute(argv[2]), "core");
        require(!installed.root.empty(), "Core runtime did not install");
        uint64_t total_bytes = 0;
        size_t checked = 0, compressed_files = 0;
        for (const auto &entry : manifest.at("files")) {
            auto name = entry.at("path").get<std::string>();
            relative_path(name);
            require(name.find("nvngx") == std::string::npos &&
                        name.find("weights") == std::string::npos,
                    "Model filename in payload");
            const auto [packed, packed_bytes] =
                resource(module, entry.at("resource").get<unsigned>());
            require(packed && packed_bytes == entry.at("packed_bytes").get<size_t>(),
                    "Embedded resource extent differs");
            size_t count = entry.at("bytes").get<size_t>();
            std::vector<uint8_t> decoded(count);
            auto algorithm = entry.at("algorithm").get<std::string>();
            if (algorithm == "raw") {
                require(count == packed_bytes, "Raw resource extent differs");
                std::memcpy(decoded.data(), packed, count);
            } else {
                require(algorithm == "xpress_huff", "Unexpected compression");
                DECOMPRESSOR_HANDLE decoder{};
                require(CreateDecompressor(COMPRESS_ALGORITHM_XPRESS_HUFF, nullptr, &decoder) !=
                            FALSE,
                        "Decompressor creation failed");
                SIZE_T written = 0;
                bool success = Decompress(decoder, packed, packed_bytes, decoded.data(),
                                          decoded.size(), &written) != FALSE;
                CloseDecompressor(decoder);
                require(success && written == count, "Resource decompression failed");
                ++compressed_files;
            }
            require(phoenix_model::weight_digest(decoded.data(), decoded.size()) ==
                        entry.at("sha256").get<std::string>(),
                    "Decoded payload checksum differs");
            if (std::strcmp(dependency_group(name), "core") == 0)
                require(file_digest(installed.root / relative_path(name)) ==
                            entry.at("sha256").get<std::string>(),
                        "Installed core checksum differs");
            else
                require(!fs::exists(installed.root / relative_path(name)),
                        "Optional vendor dependency installed during core load");
            if (name == "runtime/phoenix_native_cpp.json") {
                auto config = Json::parse(decoded.begin(), decoded.end());
                require(!config.contains("weight_source"),
                        "Local model path embedded in runtime configuration");
            }
            total_bytes += count;
            ++checked;
        }
        // Reopening a populated cache must preserve the same verified package.
        auto reopened = install(module, fs::absolute(argv[2]), "core");
        require(reopened.identity == installed.identity, "Cache identity changed on reopen");
        Json report{{"passed", true},
                    {"payload_files_verified", checked},
                    {"compressed_files_verified", compressed_files},
                    {"payload_bytes_verified", total_bytes},
                    {"core_files_installed", installed.files},
                    {"optional_vendor_dependencies_lazy", true},
                    {"model_data_included", false},
                    {"nvidia_dll_included", false},
                    {"installed_root", installed.root.u8string()}};
        std::ofstream(argv[3]) << report.dump(2) << '\n';
        std::cout << report.dump() << '\n';
        FreeLibrary(module);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
