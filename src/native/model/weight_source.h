// Validates and reads the selected model source, from a supplied DLL or supported disk layout.
// Resource parsing checks lengths and digests before a runner receives tensor payloads.

#pragma once
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#endif
#include "portable_model_io.h"
#include "registry.h"
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <limits>
#include <cstring>
#include <map>
#include "../../../third_party/nlohmann/json.hpp"

namespace phoenix_model {
using Json = nlohmann::json;
inline std::string source_path_string(const std::filesystem::path& path) {
    const auto utf8 = path.u8string();
    return {reinterpret_cast<const char*>(utf8.data()), utf8.size()};
}
struct WeightSource {
    std::filesystem::path dll;
    std::string dll_sha256;
    std::filesystem::path folder;
    bool folder_mode = false;
    std::string model_id = std::string(models::original.id);
    const Definition& model() const {
        return definition(model_id);
    }
    bool disk_model() const {
        return folder_mode || !folder.empty();
    }
    bool external() const {
        return !dll.empty() || disk_model();
    }
    Json configuration() const {
        validate();
        Json result;
        if (disk_model())
            result = {{"kind", "model_folder"}, {"path", source_path_string(folder)}};
        else if (!dll.empty())
            result = {
                {"kind", "nvidia_dll"}, {"path", source_path_string(dll)}, {"sha256", dll_sha256}};
        else
            result = {{"kind", "files"}};
        // Preserve the serialized shape of existing default configurations.
        if (model_id != models::original.id)
            result["model"] = model_id;
        return result;
    }
    static WeightSource config(const Json& value) {
        const auto kind = value.value("kind", std::string("files"));
        WeightSource result;
        result.model_id = value.value("model", std::string(models::original.id));
        (void)result.model();
        if (kind == "files") {
            if (value.contains("path") || value.contains("sha256"))
                throw std::runtime_error("File weight source does not accept DLL fields");
            result.validate();
            return result;
        }
        if (kind == "model_folder") {
            if (value.contains("sha256"))
                throw std::runtime_error("Model folder checksum belongs in model.json");
            result.folder_mode = true;
            result.folder = std::filesystem::u8path(value.at("path").get<std::string>());
            if (result.folder.empty())
                throw std::runtime_error("Model folder path is empty");
            result.validate();
            return result;
        }
        if (kind != "nvidia_dll")
            throw std::runtime_error("Unknown weight source");
        result.dll = std::filesystem::u8path(value.at("path").get<std::string>());
        result.dll_sha256 = value.at("sha256").get<std::string>();
        result.validate();
        return result;
    }
    void validate() const {
        (void)model();
        if (disk_model()) {
            if (folder.empty() || !dll.empty() || !dll_sha256.empty() || !folder.is_absolute() ||
                folder.native().find(std::filesystem::path::value_type(0)) !=
                    std::filesystem::path::string_type::npos)
                throw std::runtime_error("Model folder needs an absolute path and no DLL fields");
            return;
        }
        if (dll.empty() && dll_sha256.empty())
            return;
        if (dll.empty() || dll_sha256.size() != 64 ||
            dll_sha256.find_first_not_of("0123456789abcdef") != std::string::npos)
            throw std::runtime_error("DLL weight source needs a path and lowercase SHA-256");
    }
};
inline std::string weight_digest(const uint8_t* data, size_t bytes) {
    if (bytes > UINT32_MAX)
        throw std::runtime_error("Weight hash extent");
#ifdef _WIN32
    unsigned char hash[32]{};
    if (BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0, const_cast<PUCHAR>(data), ULONG(bytes),
                   hash, 32) < 0)
        throw std::runtime_error("Weight SHA-256 failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t iteration_index = 0; iteration_index < 32; ++iteration_index) {
        result[iteration_index * 2] = hex[hash[iteration_index] >> 4];
        result[iteration_index * 2 + 1] = hex[hash[iteration_index] & 15];
    }
    return result;
#else
    return portable_sha256(data, bytes);
#endif
}
inline std::string mapped_file_digest(const std::filesystem::path& path) {
    ReadOnlyModelFile file(path);
    return weight_digest(file.bytes().data(), file.bytes().size());
}

// Bounds are checked before every record access. The output remains packed;
// only the already authenticated catalogue supplies native offsets/padding.
struct WeightRecord {
    size_t byte_offset, bytes;
};
namespace resource_format {
// Packed little-endian resource protocol; do not map a naturally aligned C++
// struct onto it. In particular the payload begins at byte 20, not byte 24.
inline constexpr size_t length_bytes = sizeof(uint64_t);
inline constexpr size_t body_size_offset = 0;
inline constexpr size_t payload_size_offset = body_size_offset + length_bytes;
inline constexpr size_t device_offset = payload_size_offset + length_bytes;
inline constexpr size_t payload_offset = device_offset + sizeof(uint32_t);
inline constexpr size_t maximum_name_bytes = 4096;
inline constexpr uint32_t maximum_device_tag = 1;
static_assert(payload_offset == 20);
} // namespace resource_format
inline std::map<std::string, WeightRecord> resource_records(
    const uint8_t* data, size_t bytes, const Definition& model = models::original) {
    auto range = [bytes](size_t byte_offset, size_t count) {
        if (byte_offset > bytes || count > bytes - byte_offset)
            throw std::runtime_error("Truncated DLL weight resource");
    };
    auto u64 = [&](size_t byte_offset) {
        range(byte_offset, 8);
        uint64_t value;
        std::memcpy(&value, data + byte_offset, 8);
        return value;
    };
    auto u32 = [&](size_t byte_offset) {
        range(byte_offset, 4);
        uint32_t value;
        std::memcpy(&value, data + byte_offset, 4);
        return value;
    };
    if (u64(0) != bytes)
        throw std::runtime_error("DLL weight resource size mismatch");
    std::map<std::string, WeightRecord> records;
    size_t position = resource_format::length_bytes;
    while (position < bytes) {
        const auto length = u64(position);
        if (!length || length > resource_format::maximum_name_bytes)
            throw std::runtime_error("DLL weight record name extent");
        position += resource_format::length_bytes;
        range(position, size_t(length));
        std::string name(reinterpret_cast<const char*>(data + position), size_t(length));
        position += size_t(length);
        const auto body = u64(position);
        position += resource_format::length_bytes;
        if (body < resource_format::payload_offset || body > bytes - position)
            throw std::runtime_error("DLL weight record body extent");
        const auto inner = u64(position + resource_format::body_size_offset);
        const auto payload = u64(position + resource_format::payload_size_offset);
        const auto device = u32(position + resource_format::device_offset);
        if (inner != body || device > resource_format::maximum_device_tag || !payload ||
            payload > body - resource_format::payload_offset)
            throw std::runtime_error("Invalid DLL weight record");
        if (!records.emplace(name, WeightRecord{position + resource_format::payload_offset,
                                                size_t(payload)}).second)
            throw std::runtime_error("Duplicate DLL weight record");
        position += size_t(body);
    }
    if (records.size() != model.record_count)
        throw std::runtime_error("Unsupported native model record count");
    return records;
}
inline std::vector<uint8_t> reconstruct_weights(const uint8_t* data, size_t bytes,
                                                const Json& catalogue,
                                                const Definition& model = models::original) {
    const auto records = resource_records(data, bytes, model);
    const auto extent = catalogue.at("bytes").get<uint64_t>();
    if (extent > UINT32_MAX || records.size() != catalogue.at("records").size() ||
        records.size() != model.record_count || catalogue.at("unknown_gap_bytes") != 0)
        throw std::runtime_error("DLL weights do not match the complete native catalogue");
    std::vector<uint8_t> result(size_t(extent), 0);
    for (const auto& expected : catalogue.at("records")) {
        const auto record = records.find(expected.at("name").get<std::string>());
        const auto byte_offset = expected.at("offset").get<uint64_t>(),
                   count = expected.at("bytes").get<uint64_t>();
        if (record == records.end() || record->second.bytes != count || byte_offset > extent ||
            count > extent - byte_offset ||
            weight_digest(data + record->second.byte_offset, record->second.bytes) !=
                expected.at("sha256").get<std::string>())
            throw std::runtime_error("DLL weight record differs from selected model");
        std::memcpy(result.data() + size_t(byte_offset), data + record->second.byte_offset,
                    size_t(count));
    }
    if (weight_digest(result.data(), result.size()) != catalogue.at("sha256").get<std::string>())
        throw std::runtime_error("DLL reconstructed weight allocation differs");
    return result;
}
struct ResourceIdentity {
    std::string name, architecture, sha256;
};
inline ResourceIdentity resource_identity(const uint8_t* data, size_t bytes,
                                           const Definition& model) {
    return {std::string(model.resource), std::string(model.architecture), weight_digest(data, bytes)};
}
struct PackedModel {
    Json catalogue;
    std::vector<uint8_t> bytes;
    ResourceIdentity resource;
};
inline Json model_catalogue(const Registry& models = registry()) {
    Json result = Json::array();
    for (const auto& model : models.entries())
        result.push_back({{"id", model.id}, {"label", model.label},
                          {"architecture", model.architecture}, {"resource", model.resource},
                          {"implemented", model.implemented},
                          {"portable_implemented", model.implemented},
                          {"native_rtx_implemented", model.implemented && model.native.launches != 0},
                          {"unavailable_reason", model.unavailable_reason}});
    return result;
}
// The native ABI packs records lexically, each starting on a 512-byte boundary.
// Sizes and payloads come from the authenticated DLL, never a saved catalogue.
inline PackedModel pack_resource(const uint8_t* data, size_t size,
                                 const Definition& model = models::original) {
    const auto records = resource_records(data, size, model);
    PackedModel result;
    result.resource = resource_identity(data, size, model);
    result.catalogue = {
        {"records", Json::array()}, {"unknown_gap_bytes", 0}, {"source", "dll_resource"}};
    size_t extent = 0;
    for (const auto& item : records) {
        if (item.second.bytes > UINT32_MAX - extent - 511)
            throw std::runtime_error("Native packed model extent");
        result.catalogue["records"].push_back(
            {{"name", item.first},
             {"offset", extent},
             {"bytes", item.second.bytes},
             {"sha256", weight_digest(data + item.second.byte_offset, item.second.bytes)}});
        extent = (extent + item.second.bytes + 511) & ~size_t(511);
    }
    result.bytes.resize(extent, 0);
    for (const auto& record : result.catalogue["records"]) {
        const auto& source = records.at(record.at("name").get<std::string>());
        std::memcpy(result.bytes.data() + record.at("offset").get<size_t>(),
                    data + source.byte_offset, source.bytes);
    }
    result.catalogue["bytes"] = extent;
    result.catalogue["sha256"] = weight_digest(result.bytes.data(), extent);
    return result;
}
inline void validate_model_layout(const uint8_t* data, size_t bytes, const Definition& model);

template <class Read>
inline auto
with_dll_resource(const WeightSource& source,
                  const std::function<std::string(const std::filesystem::path&)>& file_digest,
                  Read read) {
    source.validate();
    const auto& model = source.model();
    require_implemented(model);
    if (source.dll.empty() || file_digest(source.dll) != source.dll_sha256)
        throw std::runtime_error("DLL weight source SHA-256 mismatch");
#ifdef _WIN32
    // Resource-only mapping never executes DllMain or NVIDIA inference code.
    struct Mapping {
        HMODULE handle;
        ~Mapping() {
            if (handle)
                FreeLibrary(handle);
        }
    } mapping{LoadLibraryExW(source.dll.c_str(), nullptr,
                             LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE | LOAD_LIBRARY_AS_IMAGE_RESOURCE)};
    if (!mapping.handle)
        throw std::runtime_error("DLL resource mapping failed");
    const std::wstring resource_name(model.resource.begin(), model.resource.end());
    const auto resource = FindResourceW(mapping.handle, resource_name.c_str(), MAKEINTRESOURCEW(10));
    if (!resource)
        throw std::runtime_error("DLL has no selected model resource: " + std::string(model.resource));
    const auto bytes = SizeofResource(mapping.handle, resource);
    const auto loaded = LoadResource(mapping.handle, resource);
    const auto data = static_cast<const uint8_t*>(LockResource(loaded));
    if (!data || !bytes)
        throw std::runtime_error("DLL weight resource unavailable");
    validate_model_layout(data, size_t(bytes), model);
    return read(data, size_t(bytes));
#else
    ReadOnlyModelFile mapping(source.dll);
    const auto resource = pe_weight_resource(mapping.bytes(), model.resource);
    // Authenticate the exact mapped bytes too, closing the hash/open race.
    if (weight_digest(mapping.bytes().data(), mapping.bytes().size()) != source.dll_sha256)
        throw std::runtime_error("Mapped DLL weight source SHA-256 mismatch");
    validate_model_layout(resource.data(), resource.size(), model);
    return read(resource.data(), resource.size());
#endif
}
inline std::vector<uint8_t>
dll_weights(const WeightSource& source, const Json& catalogue,
            const std::function<std::string(const std::filesystem::path&)>& file_digest) {
    return with_dll_resource(source, file_digest, [&](const uint8_t* data, size_t bytes) {
        return reconstruct_weights(data, bytes, catalogue, source.model());
    });
}
inline PackedModel
dll_packed_model(const WeightSource& source,
                 const std::function<std::string(const std::filesystem::path&)>& file_digest) {
    return with_dll_resource(source, file_digest, [&](const uint8_t* data, size_t bytes) {
        return pack_resource(data, bytes, source.model());
    });
}

// Architecture metadata only: the digest covers sorted record names and sizes,
// not weights. Compatible replacement values use the same fixed graph ABI.
inline constexpr const char* network37_layout_sha256 =
    models::original.layout_sha256.data();
inline void validate_model_layout(const uint8_t* data, size_t bytes,
                                  const Definition& model = models::original) {
    std::string layout;
    for (const auto& [name, record] : resource_records(data, bytes, model))
        layout += name + '\t' + std::to_string(record.bytes) + '\n';
    if (weight_digest(reinterpret_cast<const uint8_t*>(layout.data()), layout.size()) !=
        model.layout_sha256)
        throw std::runtime_error("Model record names or sizes do not match " + std::string(model.architecture));
}
inline Json folder_manifest(const std::filesystem::path& folder,
                             const Definition& model = models::original) {
    std::ifstream file(folder / "model.json", std::ios::binary);
    if (!file)
        throw std::runtime_error("Model folder has no readable model.json");
    std::string text(65537, '\0');
    file.read(text.data(), std::streamsize(text.size()));
    if (file.bad() || file.gcount() > 65536)
        throw std::runtime_error("Model manifest exceeds 64 KiB or cannot be read");
    text.resize(size_t(file.gcount()));
    auto manifest = Json::parse(text);
    if (manifest.at("schema") != "dlss5-phoenix-model/1" ||
        manifest.at("architecture").get<std::string>() != model.architecture ||
        manifest.value("model", std::string(model.id)) != model.id)
        throw std::runtime_error("Unsupported disk model schema or architecture");
    return manifest;
}
inline std::filesystem::path folder_resource_path(const std::filesystem::path& folder,
                                                  const Json& resource) {
    const auto name = resource.at("path").get<std::string>();
    const auto relative = std::filesystem::u8path(name);
    if (name.empty() || name.find('\0') != std::string::npos ||
        name.find(':') != std::string::npos || relative.has_root_path() || !relative.is_relative())
        throw std::runtime_error("Model resource needs a relative file path");
    for (const auto& part : relative)
        if (part == "..")
            throw std::runtime_error("Model resource escapes the model folder");
    const auto root = std::filesystem::canonical(folder);
    const auto path = std::filesystem::canonical(root / relative);
    auto child = path.begin();
    for (auto parent = root.begin(); parent != root.end(); ++parent, ++child)
        if (child == path.end() ||
#ifdef _WIN32
            _wcsicmp(child->c_str(), parent->c_str()) != 0
#else
            *child != *parent
#endif
        )
            throw std::runtime_error("Model resource resolves outside the model folder");
    if (child == path.end() || !std::filesystem::is_regular_file(path))
        throw std::runtime_error("Model resource is not a regular file");
    return path;
}
template <class Read> inline auto with_folder_resource(const WeightSource& source, Read read) {
    source.validate();
    const auto& model = source.model();
    require_implemented(model);
    if (source.folder.empty())
        throw std::runtime_error("Model folder is not configured");
    const auto manifest = folder_manifest(source.folder, model);
    const auto& resource = manifest.at("resource");
    const auto& extent = resource.at("bytes");
    if (!extent.is_number_integer() || extent <= 0 || extent > 512ull * 1024 * 1024)
        throw std::runtime_error("Model resource extent must be an integer from 1 to 512 MiB");
    const auto bytes = extent.get<size_t>();
    const auto checksum = resource.at("sha256").get<std::string>();
    if (checksum.size() != 64 ||
        checksum.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw std::runtime_error("Model resource needs lowercase SHA-256");
    const auto path = folder_resource_path(source.folder, resource);
    // Read-only mapping avoids another full-size host allocation. The mapping
    // and file handle live until all decoders have finished reading the bytes.
#ifdef _WIN32
    struct Mapping {
        HANDLE file = INVALID_HANDLE_VALUE, section = nullptr;
        const uint8_t* data = nullptr;
        ~Mapping() {
            if (data)
                UnmapViewOfFile(data);
            if (section)
                CloseHandle(section);
            if (file != INVALID_HANDLE_VALUE)
                CloseHandle(file);
        }
    } mapping;
    mapping.file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL, nullptr);
    LARGE_INTEGER actual{};
    if (mapping.file == INVALID_HANDLE_VALUE || !GetFileSizeEx(mapping.file, &actual) ||
        actual.QuadPart != bytes)
        throw std::runtime_error(
            "Model resource is unreadable or its byte count differs from model.json");
    mapping.section = CreateFileMappingW(mapping.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping.section)
        mapping.data =
            static_cast<const uint8_t*>(MapViewOfFile(mapping.section, FILE_MAP_READ, 0, 0, bytes));
    if (!mapping.data)
        throw std::runtime_error("Model resource read-only mapping failed");
    const auto* data = mapping.data;
#else
    ReadOnlyModelFile mapping(path);
    if (mapping.bytes().size() != bytes)
        throw std::runtime_error("Model resource byte count differs from model.json");
    const auto* data = mapping.bytes().data();
#endif
    if (weight_digest(data, bytes) != checksum)
        throw std::runtime_error("Model resource SHA-256 mismatch");
    validate_model_layout(data, bytes, model);
    return read(data, bytes);
}
template <class Read>
inline auto
with_model_resource(const WeightSource& source,
                    const std::function<std::string(const std::filesystem::path&)>& file_digest,
                    Read read) {
    if (source.disk_model())
        return with_folder_resource(source, read);
    return with_dll_resource(source, file_digest, read);
}
inline PackedModel
packed_model(const WeightSource& source,
             const std::function<std::string(const std::filesystem::path&)>& file_digest) {
    return with_model_resource(source, file_digest, [&](const uint8_t* data, size_t bytes) {
        return pack_resource(data, bytes, source.model());
    });
}
inline std::vector<uint8_t>
model_weights(const WeightSource& source, const Json& catalogue,
              const std::function<std::string(const std::filesystem::path&)>& file_digest) {
    return with_model_resource(source, file_digest, [&](const uint8_t* data, size_t bytes) {
        return reconstruct_weights(data, bytes, catalogue, source.model());
    });
}
} // namespace phoenix_model
