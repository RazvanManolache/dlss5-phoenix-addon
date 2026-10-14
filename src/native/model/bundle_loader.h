// Loads the native kernel bundle and verifies its launch/source contracts.
// Bundle discovery and model-source loading are separate so weights are not tied to packaged kernels.

#pragma once
#include "launch_plan.h"
#include "weight_source.h"
#include "generated_plan.h"
#include "../../../third_party/nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>

namespace phoenix_bundle {
using Json = nlohmann::json;
namespace fs = std::filesystem;
inline void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
inline Json json_file(const fs::path& path) {
    std::ifstream stream(path);
    require(bool(stream), "missing JSON");
    return Json::parse(stream);
}
inline bool has_barrier_policy(const Json& frame) {
    return frame.value("barriers_captured", false) ||
           frame.value("barrier_policy", std::string()) == "phoenix-native-graph/1";
}
inline std::vector<uint8_t> hex(const std::string& text) {
    require(text.size() % 2 == 0, "odd hex byte count");
    std::vector<uint8_t> result;
    auto digit = [](char hex_digit) -> unsigned {
        if (hex_digit >= '0' && hex_digit <= '9')
            return unsigned(hex_digit - '0');
        if (hex_digit >= 'a' && hex_digit <= 'f')
            return unsigned(hex_digit - 'a' + 10);
        throw std::runtime_error("invalid lowercase hex");
    };
    for (size_t iteration_index = 0; iteration_index < text.size(); iteration_index += 2)
        result.push_back(
            uint8_t(digit(text[iteration_index]) * 16 + digit(text[iteration_index + 1])));
    return result;
}
inline phoenix_launch_plan::Launch launch(const Json& source,
                                          const std::function<uint64_t(const Json&)>& descriptor) {
    phoenix_launch_plan::Launch result{};
    result.grid = source.at("grid").get<std::array<uint32_t, 3>>();
    result.block = source.at("block").get<std::array<uint32_t, 3>>();
    result.dynamic_shared_bytes = source.at("dynamic_shared_bytes").get<uint32_t>();
    result.argument_template = hex(source.at("template_hex").get<std::string>());
    require(result.argument_template.size() == source.at("argument_bytes").get<size_t>(),
            "template extent");
    for (const auto& binding : source.at("relocations")) {
        require(binding.at("bytes") == 8, "pointer slot width");
        const auto arena = binding.at("arena").get<std::string>();
        require(arena == "workspace" || arena == "weights", "unknown arena");
        const bool null = binding.at("relative_offset").is_null();
        result.bindings.push_back({binding.at("argument_offset").get<uint32_t>(),
                                   null ? phoenix_launch_plan::BindingKind::null_pointer
                                   : arena == "workspace"
                                       ? phoenix_launch_plan::BindingKind::workspace
                                       : phoenix_launch_plan::BindingKind::weights,
                                   null ? 0 : binding.at("relative_offset").get<uint64_t>()});
    }
    for (const auto& binding : source.at("descriptor_bindings")) {
        require(binding.at("bytes") == 8, "descriptor slot width");
        result.bindings.push_back({binding.at("argument_offset").get<uint32_t>(),
                                   phoenix_launch_plan::BindingKind::descriptor,
                                   binding.at("binding").is_null() ? 0 : descriptor(binding)});
    }
    return result;
}
struct Bundle {
    fs::path folder;
    Json weights, kernels, small_plan, full_plan, resize_plan, hd_plan, manifest;
    Json graph;
    phoenix_generated::PlanCache generated_plans;
    // 1920x1080. Optional: bundles exported before it existed (plan_a4 and
    // everything staged from it) must keep loading byte-for-byte as before.
    bool has_hd = false;
    bool generated = false;
    phoenix_model::WeightSource weight_source;
    std::vector<uint8_t> embedded_weights;
    // Production bundles generate geometry on demand. Legacy validation
    // bundles retain their captured frames, including recurrence preflight.
    const Json& select_frame(unsigned width, unsigned height,
                             const std::function<void(const Json&)>& validate) {
        if (generated)
            return generated_plans.select(graph, kernels, weights, width, height, validate);
        for (const auto* plan : {&small_plan, &full_plan, &resize_plan, &hd_plan}) {
            if (!plan->contains("frames"))
                continue;
            for (const auto& candidate : plan->at("frames")) {
                if (candidate.at("width") != width || candidate.at("height") != height ||
                    !has_barrier_policy(candidate))
                    continue;
                try {
                    validate(candidate);
                    return candidate;
                } catch (const std::exception&) {
                    // Older reset-only frames can lack recurrence bindings.
                }
            }
        }
        throw std::runtime_error(
            "Legacy bundle has no recurrent plan for this resolution; use a source-built native bundle");
    }
    std::vector<uint8_t> load_weights() {
        if (weight_source.external()) {
            require(!embedded_weights.empty(), "External model weights already consumed");
            return std::move(embedded_weights);
        }
        std::ifstream input(folder / "weights.bin", std::ios::binary);
        require(bool(input), "Missing complete file weights");
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }
    Bundle(fs::path path, const std::function<std::string(const fs::path&)>& digest,
           phoenix_model::WeightSource source = {})
        : folder(std::move(path)), weight_source(std::move(source)) {
        weight_source.validate();
        if (fs::exists(folder / "native_manifest.json")) {
            require(weight_source.external(),
                    "Generated native bundle requires an external model source");
            manifest = json_file(folder / "native_manifest.json");
            require(manifest.at("schema") == "phoenix-native-generated-build/1" &&
                        manifest.at("passed") == true &&
                        manifest.at("model_weights_included") == false &&
                        manifest.at("captured_plans_included") == false,
                    "Native source-build manifest");
            require(digest(folder / "native_graph.json") == manifest.at("native_graph_sha256").get<std::string>(),
                    "Native graph digest");
            graph = json_file(folder / "native_graph.json");
            const auto supported_dll = graph.at("supported_dll_sha256").get<std::string>();
            require(
                weight_source.disk_model() || weight_source.dll_sha256 == supported_dll ||
                    phoenix_model::equivalent_model_dll(weight_source.dll_sha256, supported_dll),
                "Native graph does not support this DLL version");
            auto model = phoenix_model::packed_model(weight_source, digest);
            weights = std::move(model.catalogue);
            embedded_weights = std::move(model.bytes);
            kernels = Json::object();
            for (auto kernel_record : manifest.at("kernels")) {
                const auto name = kernel_record.at("name").get<std::string>();
                require(!kernels.contains(name), "Duplicate generated kernel");
                const auto relative = fs::u8path(kernel_record.at("binary").get<std::string>());
                require(relative.is_relative() && !relative.has_root_path(),
                        "Generated kernel path");
                for (const auto& component : relative)
                    require(component != "..", "Generated kernel outside bundle");
                const auto binary = fs::absolute(folder / relative).lexically_normal();
                require(digest(binary) == kernel_record.at("binary_sha256").get<std::string>(),
                        "Generated kernel digest");
                kernel_record["binary"] = binary.string();
                kernels[name] = std::move(kernel_record);
            }
            require(kernels.size() == 41, "Generated kernel count");
            generated = true;
            return;
        }
        manifest = json_file(folder / "validation.json");
        require(manifest.at("schema") == "native-fp8-plan-y37/1", "unsupported schema");
        const std::array<const char*, 6> required{"weights.bin", "weights.json", "kernels.json",
                                                  "small.json",  "full.json",    "resize.json"};
        has_hd = manifest.at("artifacts").contains("hd.json");
        require(manifest.at("artifacts").size() == required.size() + (has_hd ? 1u : 0u),
                "artifact count");
        for (const auto* name : required) {
            const auto& spec = manifest.at("artifacts").at(name);
            if (std::string(name) == "weights.bin" && weight_source.external())
                continue;
            require(fs::file_size(folder / name) == spec.at("bytes").get<uint64_t>(),
                    "artifact size");
            require(digest(folder / name) == spec.at("sha256").get<std::string>(),
                    "artifact digest");
        }
        if (has_hd) {
            const auto& spec = manifest.at("artifacts").at("hd.json");
            require(fs::file_size(folder / "hd.json") == spec.at("bytes").get<uint64_t>(),
                    "artifact size");
            require(digest(folder / "hd.json") == spec.at("sha256").get<std::string>(),
                    "artifact digest");
        }
        weights = json_file(folder / "weights.json");
        kernels = json_file(folder / "kernels.json");
        // Authenticate the catalog above, then resolve only its file locations.
        // Kernel and model bytes retain the original digests after relocation.
        for (auto& item : kernels.items()) {
            fs::path binary = item.value().at("binary").get<std::string>();
            if (binary.is_relative())
                item.value()["binary"] = fs::absolute(folder / binary).lexically_normal().string();
        }
        small_plan = json_file(folder / "small.json");
        full_plan = json_file(folder / "full.json");
        resize_plan = json_file(folder / "resize.json");
        if (has_hd)
            hd_plan = json_file(folder / "hd.json");
        require(weights.at("records").size() == 153 && kernels.size() == 43,
                "network catalog incomplete");
        if (weights.at("unknown_gap_bytes") == 0) {
            const auto& proof = weights.at("native_capture");
            require(weights.at("native_allocation_extent_known") == true &&
                        weights.at("native_contiguous_digest_known") == true &&
                        proof.at("fenced") == true && proof.at("saved") == true &&
                        proof.at("full_payload_byte_equal") == true &&
                        proof.at("allocation_bytes") == weights.at("bytes") &&
                        proof.at("allocation_sha256") == weights.at("sha256") &&
                        proof.at("original_records_verified") == 153 &&
                        proof.at("stem_weight_offset") == 0 && proof.at("tail_bytes") == 0,
                    "native full-allocation proof incomplete");
        }
        if (weight_source.external()) {
            require(weights.at("bytes") == manifest.at("artifacts").at("weights.bin").at("bytes") &&
                        weights.at("sha256") ==
                            manifest.at("artifacts").at("weights.bin").at("sha256"),
                    "Weight catalogue/manifest mismatch");
            embedded_weights = phoenix_model::model_weights(weight_source, weights, digest);
        }
        std::vector<const Json*> captures{&small_plan, &full_plan, &resize_plan};
        if (has_hd)
            captures.push_back(&hd_plan);
        for (const auto* capture : captures) {
            require(capture->at("schema") == "native-fp8-plan-y37/1" &&
                        capture->at("frames").size() == 32,
                    "frame count/schema");
            uint32_t number = 0;
            for (const auto& frame : capture->at("frames")) {
                require(frame.at("frame") == ++number && frame.at("launches").size() == 156,
                        "launch/frame order");
                const auto& launches = frame.at("launches");
                require(launches[0].at("name") == "cc_cb_clear" &&
                            launches[155].at("name") == "cg2r_copy_kernel",
                        "boundary symbols");
                uint32_t index = 0;
                for (const auto& source : launches) {
                    require(source.at("index") == index, "launch index");
                    if (index > 0 && index < 155) {
                        const auto& spec = kernels.at(source.at("name").get<std::string>());
                        require(spec.at("parameter_bytes") == source.at("argument_bytes") &&
                                    spec.at("block") == source.at("block"),
                                "candidate ABI");
                    }
                    ++index;
                }
                if (frame.at("barriers_captured") == true) {
                    const std::array<unsigned, 15> positions{1,  2,  6,   10,  16,  24,  57, 58,
                                                             98, 99, 132, 140, 146, 150, 154};
                    const auto& barriers = frame.at("neural_region_barriers");
                    require(frame.at("barriers_captured") == true &&
                                barriers.size() == positions.size(),
                            "barrier capture required");
                    for (size_t iteration_index = 0; iteration_index < positions.size();
                         ++iteration_index)
                        require(barriers[iteration_index].at("after_launch") ==
                                        positions[iteration_index] &&
                                    barriers[iteration_index].at("resource") == 0 &&
                                    barriers[iteration_index].at("before") == 8 &&
                                    barriers[iteration_index].at("after") == 8,
                                "native barrier mismatch");
                }
            }
        }
    }
};
} // namespace network37
