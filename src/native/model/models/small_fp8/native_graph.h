#pragma once
#include "../../registry.h"
#include "../../window_kernel_abi.h"
#include "../../../../../third_party/nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace phoenix_model::models {
namespace small_native_detail {
using Json = nlohmann::json;

inline void require(bool valid, const char* message) {
    if (!valid)
        throw std::runtime_error(message);
}

// One-based launch indices in the authenticated original ABI graph, not small
// block IDs. Shortened stages keep their entry and exit. Split-window blocks
// span several kernels, so retain each selected block's complete range.
inline std::vector<unsigned> retained_launches() {
    std::vector<unsigned> launches;
    auto append_range = [&](unsigned first, unsigned last) {
        for (unsigned launch = first; launch <= last; ++launch)
            launches.push_back(launch);
    };
    auto append_stage = [&](unsigned entry, unsigned exit) {
        launches.push_back(entry);
        launches.push_back(exit);
    };

    launches.push_back(1);   // Input stem.
    append_stage(2, 5);      // 32-channel encoder.
    append_stage(6, 9);      // 64-channel encoder.
    append_stage(10, 15);    // 128-channel encoder.
    append_stage(16, 23);    // 256-channel encoder.
    append_range(24, 27);    // First 512-channel split-window block.
    append_range(52, 56);    // Last split-window block and pooled projection.
    append_range(57, 98);    // Token repacks and all eight global transformers.
    launches.push_back(99);  // 1024-to-512 decoder input upsampling.
    append_range(100, 103);  // First 512-channel decoder block.
    append_range(128, 131);  // Last split-window decoder block.
    append_stage(132, 139);  // 256-channel decoder.
    append_stage(140, 145);  // 128-channel decoder.
    append_stage(146, 149);  // 64-channel decoder.
    append_stage(150, 153);  // 32-channel decoder.
    launches.push_back(154); // Output head and history blend.
    return launches;
}

inline bool has_prefix(const std::string& text, const char* prefix) {
    return text.rfind(prefix, 0) == 0;
}

inline bool is_weight_target(const Json& target) {
    return target.is_string() && has_prefix(target.get<std::string>(), "weight:block");
}

inline unsigned weight_block(const std::string& target) {
    constexpr size_t block_number_offset = sizeof("weight:block") - 1;
    return unsigned(std::stoul(target.substr(block_number_offset)));
}

using WindowShift = window_abi::Shift;

// Native kernels express shifts in four-pixel units encoded as unsigned
// parameter words. The portable graph uses window units instead.
inline constexpr uint32_t negative_shift = uint32_t(-4);
inline constexpr std::array<WindowShift, 4> window_shifts{
    {{0, 0}, {negative_shift, negative_shift}, {negative_shift, 0}, {0, negative_shift}}};
inline constexpr unsigned blocks_per_short_stage = 2;
inline constexpr unsigned first_fused_decoder_block = 22;
inline constexpr unsigned first_split_encoder_block = 9;
inline constexpr unsigned last_split_encoder_block = 10;
inline constexpr unsigned first_split_decoder_block = 20;

inline std::string shifted_grid(const std::string& extent, uint32_t shift) {
    return "ceil8(" + extent + "+" + (shift ? "4" : "0") + ")*1";
}

inline void set_shift(Json& scalars, unsigned x_byte_offset, WindowShift shift) {
    scalars[std::to_string(x_byte_offset)] = shift.x;
    scalars[std::to_string(x_byte_offset + sizeof(uint32_t))] = shift.y;
}

inline void patch_fused_window(Json& layer, unsigned block) {
    const auto kernel = layer.at("kernel").get<std::string>();
    const bool decoder = block >= first_fused_decoder_block;
    const unsigned phase = decoder ? blocks_per_short_stage +
        (block - first_fused_decoder_block) % blocks_per_short_stage :
        (block - 1) % blocks_per_short_stage;
    const unsigned previous_phase = phase % 2 ? phase - 1 : 0;
    const auto shift = window_shifts[phase];

    // 32-channel kernels have a compact parameter layout. Wider fused kernels
    // add an earlier pointer pair; all offsets here are bytes in the kernel ABI.
    const auto offsets = kernel.find("_1h_32_") != std::string::npos ?
                             window_abi::compact : window_abi::wide;
    auto& scalars = layer["scalars"];
    set_shift(scalars, offsets.current, shift);
    set_shift(scalars, offsets.previous, window_shifts[previous_phase]);
    const auto height = scalars.at(std::to_string(offsets.height)).get<std::string>();
    const auto width = scalars.at(std::to_string(offsets.width)).get<std::string>();
    layer["grid"][0] = shifted_grid(width, shift.x);
    layer["grid"][1] = shifted_grid(height, shift.y);
}

inline void patch_split_window(Json& layer, unsigned block) {
    const auto kernel = layer.at("kernel").get<std::string>();
    // Encoder phases are 0,1. Decoder phases are 2,3, matching the recovered
    // host rule (stage_block_count + block_index) % 4.
    const unsigned phase = block <= last_split_encoder_block ? block - first_split_encoder_block :
        block - first_split_decoder_block + blocks_per_short_stage;
    const auto shift = window_shifts[phase];
    auto& scalars = layer["scalars"];
    if (kernel.find("_qkv_") != std::string::npos) {
        set_shift(scalars, window_abi::split_qkv_shift, shift);
        layer["grid"][0] = shifted_grid("W5", shift.x);
        layer["grid"][1] = shifted_grid("H5", shift.y);
    } else if (kernel.find("_proj_pool_") != std::string::npos) {
        set_shift(scalars, window_abi::split_pool_shift, shift);
    } else if (has_prefix(kernel, "cc_split_swin_16h_proj_512_")) {
        set_shift(scalars, window_abi::split_projection_shift, shift);
    }
}

// Own remapping state for one specialization. The original graph remains
// immutable, including its tensor names, shared kernel ABI and authentication.
class GraphSpecializer {
    const Json& original_;
    const std::vector<unsigned> retained_ = retained_launches();
    const std::set<unsigned> retained_set_{retained_.begin(), retained_.end()};
    std::map<unsigned, unsigned> launch_indices_;
    std::map<unsigned, unsigned> weight_blocks_;
    std::map<std::string, Json> removed_edges_;

    void map_retained_indices() {
        for (unsigned index = 0; index < retained_.size(); ++index)
            launch_indices_[retained_[index]] = index + 1;

        // A split block mentions the same block in several launches. Assign
        // small-model block IDs only once, in execution order.
        unsigned next_block = 0;
        for (unsigned launch : retained_)
            for (const auto& binding :
                 original_.at("layers").at(launch - 1).at("bindings").items()) {
                const auto& target = binding.value().at("target");
                if (!is_weight_target(target))
                    continue;
                const auto block = weight_block(target.get<std::string>());
                if (!weight_blocks_.count(block))
                    weight_blocks_[block] = next_block++;
            }
        require(next_block == 31 && retained_.size() == 78, "Small native graph block count");
    }

    void collect_removed_edges() {
        for (const auto& layer : original_.at("layers")) {
            if (retained_set_.count(layer.at("index").get<unsigned>()))
                continue;
            Json feature_predecessor, publication_predecessor;
            for (const auto& binding : layer.at("bindings").items()) {
                const auto role = binding.value().at("role").get<std::string>();
                if (role == "input")
                    feature_predecessor = binding.value().at("target");
                if (has_prefix(role, "wait"))
                    publication_predecessor = binding.value().at("target");
            }
            require(feature_predecessor.is_string(),
                    "Removed native block has no feature predecessor");
            // Reconnect data and synchronization independently. A removed
            // publication must not leave a consumer waiting on stale storage.
            for (const auto& binding : layer.at("bindings").items()) {
                const auto& target = binding.value().at("target");
                const auto role = binding.value().at("role").get<std::string>();
                if (!target.is_string())
                    continue;
                if (has_prefix(role, "output"))
                    removed_edges_[target.get<std::string>()] = feature_predecessor;
                if (has_prefix(role, "publication"))
                    removed_edges_[target.get<std::string>()] = publication_predecessor;
            }
        }
    }

    Json resolve_target(Json target) const {
        for (unsigned hops = 0;
             target.is_string() && removed_edges_.count(target.get<std::string>()); ++hops) {
            require(hops < 154, "Native graph edge cycle");
            target = removed_edges_.at(target.get<std::string>());
        }
        if (is_weight_target(target)) {
            const auto name = target.get<std::string>();
            const auto field_separator = name.find('.', sizeof("weight:block") - 1);
            target = "weight:block" + std::to_string(weight_blocks_.at(weight_block(name))) +
                     name.substr(field_separator);
        }
        return target;
    }

    static std::optional<unsigned> model_block(const Json& layer) {
        for (const auto& binding : layer.at("bindings").items()) {
            const auto& target = binding.value().at("target");
            if (is_weight_target(target))
                return weight_block(target.get<std::string>());
        }
        return std::nullopt; // Layout-only kernels have no learned weights.
    }

    Json rewrite_layer(unsigned source_index) const {
        auto layer = original_.at("layers").at(source_index - 1);
        layer["index"] = launch_indices_.at(source_index);
        for (auto& binding : layer["bindings"].items())
            binding.value()["target"] = resolve_target(binding.value().at("target"));

        const auto block = model_block(layer);
        if (!block)
            return layer;
        const auto kernel = layer.at("kernel").get<std::string>();
        if (has_prefix(kernel, "cc_tinlayout_fused_swin_"))
            patch_fused_window(layer, *block);
        if (has_prefix(kernel, "cc_split_swin_16h_"))
            patch_split_window(layer, *block);
        return layer;
    }

    void rebuild_tensor_lifetimes(Json& graph) const {
        graph["tensors"] = Json::object();
        for (const auto& entry : original_.at("tensors").items()) {
            const auto producer = entry.value().at("producer").get<unsigned>();
            if (!retained_set_.count(producer))
                continue;
            auto tensor = entry.value();
            tensor["producer"] = launch_indices_.at(producer);
            tensor["last_use"] = launch_indices_.at(producer);
            graph["tensors"][entry.key()] = std::move(tensor);
        }
        // Shortening stages changes the last reader of feature tensors and
        // publication counters. Recompute every lifetime before arena planning.
        for (const auto& layer : graph.at("layers"))
            for (const auto& binding : layer.at("bindings").items()) {
                const auto& target = binding.value().at("target");
                if (!target.is_string() || !has_prefix(target.get<std::string>(), "tensor:"))
                    continue;
                auto& tensor = graph["tensors"].at(target.get<std::string>());
                tensor["last_use"] = std::max(tensor.at("last_use").get<unsigned>(),
                                              layer.at("index").get<unsigned>());
            }
    }

  public:
    explicit GraphSpecializer(const Json& original) : original_(original) {}

    Json build() {
        require(
            original_.at("schema") == "phoenix-native-graph/1" &&
                original_.value("model_architecture", std::string(models::original.architecture)) ==
                    models::original.architecture &&
                original_.at("layers").size() == 154,
            "Small native graph requires the original source-built kernel ABI graph");
        map_retained_indices();
        collect_removed_edges();

        Json graph = original_;
        graph["model_architecture"] = small_fp8.architecture;
        graph["supported_dll_sha256"] = small_fp8.reference_dll_sha256;
        graph["reference_model_resource"] = {{"name", small_fp8.resource},
                                             {"sha256", small_fp8.reference_resource_sha256}};
        graph["layers"] = Json::array();
        // Preserve the proven read-after-write boundaries of the 78-launch ABI.
        graph["barriers_after"] = {1, 2, 4, 6, 8, 10, 19, 20, 60, 61, 70, 72, 74, 76, 78};
        for (unsigned source_index : retained_)
            graph["layers"].push_back(rewrite_layer(source_index));
        rebuild_tensor_lifetimes(graph);
        return graph;
    }
};
} // namespace small_native_detail

// Specialize the authenticated original kernel ABI graph in memory. No model
// bytes, captured launch packets or changes to the shared bundle are required.
inline nlohmann::json small_native_graph(const nlohmann::json& original) {
    return small_native_detail::GraphSpecializer(original).build();
}
} // namespace phoenix_model::models
