#pragma once
#include "weight_source.h"
#include <algorithm>
#include <mutex>

#ifdef _WIN32
namespace phoenix_model {
struct StreamedPackedModel {
    Json catalogue;
    ResourceIdentity resource;
    WeightView bytes;
};

// Authenticate the packed native layout while retaining only record metadata.
// The file stays open without write sharing until the last reader is released.
inline StreamedPackedModel streamed_packed_model(const WeightSource& source,
    const std::function<std::string(const std::filesystem::path&)>& digest,
    const Json* expected_catalogue = nullptr) {
    struct Region { size_t source, destination, bytes; };
    struct Storage {
        std::shared_ptr<ReadOnlyModelFile> file;
        std::vector<Region> regions;
        std::mutex reading;
    };
    auto storage = std::make_shared<Storage>();
    storage->file = open_streamed_model_resource(source, digest);
    const auto& model = source.model();
    const auto records = resource_records_with_reader(storage->file->resource_size(),
        [&](size_t offset, void* output, size_t count) { storage->file->read_resource(offset, output, count); }, model);
    validate_model_record_layout(records, model);
    StreamedPackedModel result;
    result.resource = {std::string(model.resource), std::string(model.architecture),
        storage->file->resource_digest(0, storage->file->resource_size())};
    result.catalogue = expected_catalogue ? *expected_catalogue :
        Json{{"records", Json::array()}, {"unknown_gap_bytes", 0}, {"source", "dll_resource"}};
    size_t extent = 0;
    if (!expected_catalogue) {
        for (const auto& [name, record] : records) {
            if (record.bytes > UINT32_MAX - extent - 511)
                throw std::runtime_error("Native packed model extent");
            result.catalogue["records"].push_back({{"name", name}, {"offset", extent},
                {"bytes", record.bytes}, {"sha256", storage->file->resource_digest(record.byte_offset, record.bytes)}});
            extent = (extent + record.bytes + 511) & ~size_t(511);
        }
        result.catalogue["bytes"] = extent;
    } else {
        const auto bytes = result.catalogue.at("bytes").get<uint64_t>();
        if (!bytes || bytes > UINT32_MAX) throw std::runtime_error("Native packed model extent");
        extent = size_t(bytes);
    }
    if (result.catalogue.at("records").size() != records.size() ||
        result.catalogue.at("unknown_gap_bytes") != 0)
        throw std::runtime_error("DLL weights do not match the complete native catalogue");
    std::map<std::string, bool> seen;
    for (const auto& expected : result.catalogue.at("records")) {
        const auto name = expected.at("name").get<std::string>();
        const auto found = records.find(name);
        const auto offset = expected.at("offset").get<uint64_t>(), count = expected.at("bytes").get<uint64_t>();
        if (!seen.emplace(name, true).second || found == records.end() || found->second.bytes != count ||
            offset > extent || count > extent - offset ||
            (expected_catalogue && storage->file->resource_digest(found->second.byte_offset, found->second.bytes) !=
                expected.at("sha256").get<std::string>()))
            throw std::runtime_error("DLL weight record differs from selected model");
        storage->regions.push_back({found->second.byte_offset, size_t(offset), size_t(count)});
    }
    std::sort(storage->regions.begin(), storage->regions.end(),
        [](const Region& left, const Region& right) { return left.destination < right.destination; });
    size_t end = 0;
    for (const auto& region : storage->regions) {
        if (region.destination < end) throw std::runtime_error("Overlapping native weight records");
        end = region.destination + region.bytes;
    }
    result.bytes = WeightView(extent, storage.get(), [storage](size_t offset, void* output, size_t count) {
        std::lock_guard lock(storage->reading);
        // Padding is part of the packed hash. ReadFile receives bounded CPU
        // scratch storage, never a GPU BAR mapping supplied by the caller.
        std::memset(output, 0, count);
        std::array<uint8_t, 64 * 1024> scratch{};
        auto* destination = static_cast<uint8_t*>(output);
        const auto end = offset + count;
        auto current = std::lower_bound(storage->regions.begin(), storage->regions.end(), offset,
            [](const Region& region, size_t position) { return region.destination + region.bytes <= position; });
        for (; current != storage->regions.end() && current->destination < end; ++current) {
            size_t position = std::max(offset, current->destination);
            const auto limit = std::min(end, current->destination + current->bytes);
            while (position < limit) {
                const auto bytes = std::min(scratch.size(), limit - position);
                storage->file->read_resource(current->source + position - current->destination, scratch.data(), bytes);
                std::memcpy(destination + position - offset, scratch.data(), bytes);
                position += bytes;
            }
        }
    });
    const auto packed_digest = windows_reader_digest(extent,
        [&](size_t offset, void* output, size_t count) { result.bytes.copy_to(offset, output, count); });
    if (expected_catalogue && packed_digest != result.catalogue.at("sha256").get<std::string>())
        throw std::runtime_error("DLL reconstructed weight allocation differs");
    result.catalogue["sha256"] = packed_digest;
    result.bytes = result.bytes.authenticated(packed_digest);
    return result;
}
} // namespace phoenix_model
#endif
