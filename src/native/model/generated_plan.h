// Static native launch metadata emitted from the validated model graph.
// Numbers describe binding offsets and geometry, not embedded model weights.

#pragma once
#include "../../../third_party/nlohmann/json.hpp"
#include "../support/controls.h"
#include <map>
#include <regex>
#include <algorithm>
#include <vector>

namespace phoenix_generated {
using Json = nlohmann::json;
inline uint64_t align(uint64_t value, uint64_t boundary) {
    return (value + boundary - 1) / boundary * boundary;
}
inline void check(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
inline std::string hex_bytes(const uint8_t* data, size_t count) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result(count * 2, '0');
    for (size_t iteration_index = 0; iteration_index < count; ++iteration_index) {
        result[iteration_index * 2] = digits[data[iteration_index] >> 4];
        result[iteration_index * 2 + 1] = digits[data[iteration_index] & 15];
    }
    return result;
}
struct Geometry {
    std::map<std::string, uint32_t> values;
    Geometry(unsigned width, unsigned height) {
        check(width && height && width <= 16384 && height <= 16384, "Generated frame extent");
        // Native tiled ABI: six reductions, with a shifted-window margin when
        // both padded axes land on a complete 256-pixel cell.
        unsigned padded_width = unsigned(std::max<uint64_t>(384, align(width, 128)));
        unsigned padded_height = unsigned(std::max<uint64_t>(384, align(height, 128)));
        if (padded_width % 256 == 0 && padded_height % 256 == 0)
            padded_width += 64;
        values = {{"width", width},
                  {"height", height},
                  {"LH", unsigned(align(padded_height / 64, 4))},
                  {"LW", unsigned(align(padded_width / 64, 4))}};
        for (unsigned level_index = 0; level_index < 6; ++level_index) {
            values["H" + std::to_string(level_index)] = padded_height >> level_index;
            values["W" + std::to_string(level_index)] = padded_width >> level_index;
        }
        values["H5"] = unsigned(align(padded_height / 32, 4));
        values["W5"] = unsigned(align(padded_width / 32, 4));
        values["T"] = values.at("LH") * values.at("LW");
    }
    uint32_t evaluate(const Json& value) const {
        if (value.is_number_unsigned() || value.is_number_integer())
            return value.get<uint32_t>();
        const auto expression = value.get<std::string>();
        auto lookup_iterator = values.find(expression);
        if (lookup_iterator != values.end())
            return lookup_iterator->second;
        std::smatch match;
        if (std::regex_match(expression, match,
                             std::regex(R"(ceil8\(([A-Z][A-Z0-9]*)\+([04])\)\*([124]))")))
            return unsigned((values.at(match[1]) + std::stoul(match[2]) + 7) / 8 *
                            std::stoul(match[3]));
        if (std::regex_match(expression, match, std::regex(R"(up128\(T\)/([0-9]+))")))
            return unsigned(align(values.at("T"), 128) / std::stoul(match[1]));
        if (expression == "ceil256(T)")
            return unsigned(align(values.at("T"), 256) / 256);
        throw std::runtime_error("Unknown native geometry expression: " + expression);
    }
};
struct Tensor {
    std::string name;
    unsigned begin, end;
    uint64_t bytes, offset = 0;
};
// A chained group can overlap on the GPU. All its intermediates stay live
// until the group's barrier, even when the next dispatch has already begun.
// Reserve the counter prefix, then place tensors by barrier-expanded live
// ranges. A chained group may overlap internally, so reuse starts only after
// its barrier. Argument packets contain relocations, not absolute addresses.
inline Json frame(const Json& graph, const Json& kernels, const Json& weights, unsigned width,
                  unsigned height) {
    check(graph.at("schema") == "phoenix-native-graph/1", "Generated graph schema");
    Geometry geometry(width, height);
    std::map<std::string, uint64_t> offsets;
    for (const auto& weight_record : weights.at("records"))
        offsets["weight:" + weight_record.at("name").get<std::string>()] =
            weight_record.at("offset");
    const auto& layers = graph.at("layers");
    check(layers.size() == 154, "Generated graph extent");
    uint64_t counter_bytes = 0;
    for (const auto& layer : layers) {
        uint64_t cells = 1;
        for (const auto& axis : layer.at("grid"))
            cells *= geometry.evaluate(axis);
        const uint64_t size = align(std::max(cells, uint64_t(geometry.values.at("T"))) * 4, 512);
        for (const auto& binding : layer.at("bindings").items()) {
            const auto& target = binding.value().at("target");
            if (target.is_null())
                continue;
            const auto name = target.get<std::string>();
            if (name.rfind("counter:", 0) == 0 && !offsets.count(name)) {
                offsets[name] = counter_bytes;
                counter_bytes += size;
            }
        }
    }
    check(counter_bytes && counter_bytes < UINT32_MAX, "Generated synchronization extent");
    auto group_end = [&](unsigned launch) {
        for (const auto& position : graph.at("barriers_after"))
            if (position.get<unsigned>() - 1 >= launch)
                return position.get<unsigned>() - 1;
        return 154u;
    };
    auto group_begin = [&](unsigned launch) {
        unsigned start = 1;
        for (const auto& position : graph.at("barriers_after")) {
            const auto end = position.get<unsigned>() - 1;
            if (end >= launch)
                break;
            start = end + 1;
        }
        return start;
    };
    std::vector<Tensor> tensors;
    for (const auto& item : graph.at("tensors").items()) {
        const auto& spec = item.value();
        uint64_t bytes = 1;
        for (const auto& dim : spec.at("shape"))
            bytes *= geometry.evaluate(dim);
        tensors.push_back({item.key(), group_begin(spec.at("producer")),
                           group_end(spec.at("last_use")), align(bytes, 512)});
    }
    std::sort(
        tensors.begin(), tensors.end(), [](const Tensor& left_tensor, const Tensor& right_tensor) {
            if (left_tensor.name == "tensor:001.output_full_stem_skip")
                return right_tensor.name != left_tensor.name;
            if (right_tensor.name == "tensor:001.output_full_stem_skip")
                return false;
            return left_tensor.begin != right_tensor.begin ? left_tensor.begin < right_tensor.begin
                                                           : left_tensor.name < right_tensor.name;
        });
    uint64_t extent = counter_bytes;
    for (size_t iteration_index = 0; iteration_index < tensors.size(); ++iteration_index) {
        auto& tensor = tensors[iteration_index];
        uint64_t candidate = counter_bytes;
        std::vector<const Tensor*> live;
        for (size_t inner_index = 0; inner_index < iteration_index; ++inner_index)
            if (tensors[inner_index].end >= tensor.begin)
                live.push_back(&tensors[inner_index]);
        std::sort(live.begin(), live.end(), [](auto left_tensor, auto right_tensor) {
            return left_tensor->offset < right_tensor->offset;
        });
        for (const auto* previous : live) {
            if (candidate + tensor.bytes <= previous->offset)
                break;
            candidate = std::max(candidate, previous->offset + previous->bytes);
        }
        tensor.offset = candidate;
        offsets[tensor.name] = candidate;
        extent = std::max(extent, candidate + tensor.bytes);
    }
    // Stem skip must immediately follow the counter prefix (clear UAV extent).
    check(offsets.at("tensor:001.output_full_stem_skip") == counter_bytes, "Generated stem prefix");
    Json result = {
        {"width", width},
        {"height", height},
        {"frame", 1},
        {"barriers_captured", false},
        {"barrier_policy", "phoenix-native-graph/1"},
        {"arenas",
         {{"workspace", {{"bytes", extent}}}, {"weights", {{"bytes", weights.at("bytes")}}}}},
        {"launches", Json::array()},
        {"neural_region_barriers", Json::array()},
        {"allocation", Json::array()}};
    auto emit = [&](unsigned index, const std::string& name, const Json& grid, const Json& block,
                    unsigned bytes, const Json& bindings, const Json& descriptors,
                    std::vector<uint8_t> raw) {
        Json launch = {{"index", index},
                       {"name", name},
                       {"grid", grid},
                       {"block", block},
                       {"argument_bytes", bytes},
                       {"dynamic_shared_bytes", 0},
                       {"template_hex", hex_bytes(raw.data(), raw.size())},
                       {"relocations", Json::array()},
                       {"descriptor_bindings", Json::array()},
                       {"scalar_u32_le", Json::array()}};
        std::vector<bool> covered(bytes, false);
        for (const auto& binding : bindings.items()) {
            const auto argument_offset = unsigned(std::stoul(binding.key()));
            const auto& binding_spec = binding.value();
            const auto& target = binding_spec.at("target");
            const auto target_name = target.is_null() ? std::string() : target.get<std::string>();
            Json relocation = {
                {"argument_offset", argument_offset},
                {"bytes", 8},
                {"role", binding_spec.at("role")},
                {"arena", target_name.rfind("weight:", 0) == 0 ? "weights" : "workspace"},
                {"relative_offset", target.is_null() ? Json() : Json(offsets.at(target_name))}};
            launch["relocations"].push_back(relocation);
            for (unsigned inner_index = 0; inner_index < 8; ++inner_index)
                covered.at(argument_offset + inner_index) = true;
        }
        for (const auto& descriptor : descriptors.items()) {
            const auto argument_offset = unsigned(std::stoul(descriptor.key()));
            const auto& descriptor_spec = descriptor.value();
            launch["descriptor_bindings"].push_back(
                {{"argument_offset", argument_offset},
                 {"bytes", 8},
                 {"role", descriptor_spec.at("role")},
                 {"binding", descriptor_spec.at("binding")},
                 {"encoding", "low32_descriptor_high32_sampler"}});
            for (unsigned inner_index = 0; inner_index < 8; ++inner_index)
                covered.at(argument_offset + inner_index) = true;
        }
        for (unsigned argument_offset = 0; argument_offset < bytes; argument_offset += 4)
            if (!covered.at(argument_offset))
                launch["scalar_u32_le"].push_back(
                    {{"argument_offset", argument_offset},
                     {"bits_hex", hex_bytes(raw.data() + argument_offset, 4)}});
        result["launches"].push_back(std::move(launch));
    };
    offsets["counter:clear"] = 0;
    std::vector<uint8_t> clear(16);
    const auto words = uint32_t(counter_bytes / 4);
    std::memcpy(clear.data() + 8, &words, 4);
    emit(0, "cc_cb_clear", Json::array({unsigned(align(words, 256) / 256), 1, 1}),
         Json::array({256, 1, 1}), 16,
         {{"0", {{"role", "synchronization_arena"}, {"target", "counter:clear"}}}}, Json::object(),
         clear);
    const auto controls = phoenix_addon::resolve_controls({}, width, height, true, true, 1);
    for (const auto& layer : layers) {
        const std::string name = layer.at("kernel");
        const auto& kernel = kernels.at(name);
        const auto bytes = kernel.at("parameter_bytes").get<unsigned>();
        std::vector<uint8_t> raw(bytes, 0);
        for (const auto& scalar : layer.at("scalars").items()) {
            auto value = geometry.evaluate(scalar.value());
            std::memcpy(raw.data() + std::stoul(scalar.key()), &value, 4);
        }
        const unsigned index = layer.at("index");
        if (index == 1)
            phoenix_addon::patch_stem(raw.data(), raw.size(), controls);
        if (index == 154)
            phoenix_addon::patch_head(raw.data(), raw.size(), controls);
        Json grid = Json::array();
        for (const auto& axis : layer.at("grid"))
            grid.push_back(geometry.evaluate(axis));
        emit(index, name, grid, kernel.at("block"), bytes, layer.at("bindings"),
             layer.at("descriptors"), raw);
    }
    emit(155, "cg2r_copy_kernel",
         Json::array({unsigned(align(width, 32) / 32), unsigned(align(height, 8) / 8), 1}),
         Json::array({32, 8, 1}), 72, Json::object(),
         {{"0", {{"role", "post_copy_source_texture"}, {"binding", "output"}}},
          {"8", {{"role", "post_copy_destination_surface"}, {"binding", "destination"}}}},
         std::vector<uint8_t>(72));
    for (const auto& barrier_position : graph.at("barriers_after"))
        result["neural_region_barriers"].push_back(
            {{"after_launch", barrier_position}, {"resource", 0}, {"before", 8}, {"after", 8}});
    for (const auto& tensor : tensors)
        result["allocation"].push_back({{"tensor", tensor.name},
                                        {"offset", tensor.offset},
                                        {"bytes", tensor.bytes},
                                        {"first_group", tensor.begin},
                                        {"last_group", tensor.end}});
    return result;
}
} // namespace phoenix_generated
