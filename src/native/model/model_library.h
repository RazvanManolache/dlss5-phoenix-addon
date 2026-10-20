// Shared native model discovery. The downloader publishes its index only after
// conversion and verification; each runner independently authenticates weights.
#pragma once
#include "weight_source.h"
#include <chrono>
#include <set>

namespace phoenix_model {
struct InstalledModel {
    std::string id, label, model_id;
    std::filesystem::path folder;
};

inline Json library_json(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Shared model metadata is unavailable");
    std::string text(65537, '\0');
    input.read(text.data(), std::streamsize(text.size()));
    if (input.bad() || input.gcount() > 65536)
        throw std::runtime_error("Shared model metadata exceeds 64 KiB");
    text.resize(size_t(input.gcount()));
    return Json::parse(text);
}

inline std::vector<InstalledModel> discover_models(const std::filesystem::path& root) {
    std::vector<InstalledModel> result;
    if (root.empty() || !root.is_absolute()) return result;
    try {
        const auto index = library_json(root / "model-library.json");
        if (index.at("schema") != "phoenix-model-library/1" || !index.at("models").is_array() ||
            index.at("models").size() > 32) return result;
        std::set<std::string> ids;
        for (const auto& entry : index.at("models")) {
            try {
                const auto id = entry.at("id").get<std::string>();
                const auto label = entry.at("label").get<std::string>();
                if (id.empty() || id.size() > 128 ||
                    id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-") != std::string::npos ||
                    label.empty() || label.size() > 256 || label.find_first_of("\r\n\0", 0, 3) != std::string::npos ||
                    !ids.insert(id).second) continue;
                const auto& model = definition(entry.at("model").get<std::string>());
                if (!model.implemented) continue;
                const auto manifest_path = folder_resource_path(root, {{"path", entry.at("manifest")}});
                if (std::filesystem::file_size(manifest_path) > 65536 ||
                    mapped_file_digest(manifest_path) != entry.at("manifest_sha256").get<std::string>()) continue;
                const auto folder = manifest_path.parent_path();
                const auto manifest = folder_manifest(folder, model);
                const auto& resource = manifest.at("resource");
                const auto bytes = resource.at("bytes").get<uint64_t>();
                const auto digest = resource.at("sha256").get<std::string>();
                if (!bytes || bytes > 512ull * 1024 * 1024 || digest.size() != 64 ||
                    digest.find_first_not_of("0123456789abcdef") != std::string::npos) continue;
                const auto weights = folder_resource_path(folder, resource);
                if (std::filesystem::file_size(weights) != bytes) continue;
                result.push_back({id, label, std::string(model.id), folder});
            } catch (const std::exception&) {
                // Incomplete, moved, unsupported or malformed packages never
                // create selectable entries. The active source is not changed.
            }
        }
    } catch (const std::exception&) {
    }
    return result;
}

// Small metadata polling only. Full resource SHA/layout checks occur on load,
// away from drawing; never hash hundreds of MB on each overlay frame.
inline const std::vector<InstalledModel>& available_models(const std::filesystem::path& root) {
    struct Cache {
        std::filesystem::path root;
        std::chrono::steady_clock::time_point next{};
        std::vector<InstalledModel> entries;
    };
    static thread_local Cache cache;
    const auto now = std::chrono::steady_clock::now();
    if (cache.root != root || now >= cache.next) {
        cache.entries = discover_models(root);
        cache.root = root;
        cache.next = now + std::chrono::seconds(2);
    }
    return cache.entries;
}
} // namespace phoenix_model
