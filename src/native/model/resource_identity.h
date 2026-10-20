#pragma once
#include "weight_source.h"

namespace phoenix_model {
// NODIX Manager 1.3.7 distributes OpenNR's inspected FP16 compatibility build.
// Both resource hashes are still checked independently for the selected model.
inline constexpr std::string_view compat_fp16_310_9_16_sha256 =
    "cdfa6c1ec784867182164c7836f324f087b5c9f374d0c9fe46f54999feabf3ed";
// The picker can feed NVIDIA-code execution routes too. Resource equivalence
// alone is not approval to execute an arbitrary container's code.
inline bool known_model_container(std::string_view sha256) {
    for (const auto& model : registry().entries())
        if (!model.reference_dll_sha256.empty() && model.reference_dll_sha256 == sha256)
            return true;
    // Previously accepted local patched edition; preserve its existing picker support.
    return sha256 == "8270b350cd82de5ce89806872cdd6b6a9249b80836b91bbeb3573470744cc206" ||
           sha256 == compat_fp16_310_9_16_sha256;
}

inline bool resource_sha256_valid(std::string_view sha) {
    return sha.size() == 64 && sha.find_first_not_of("0123456789abcdef") == std::string_view::npos;
}

inline void require_resource_identity(const Definition& model, const ResourceIdentity& actual,
                                      std::string_view name, std::string_view sha256) {
    if (name != model.resource || actual.name != name || actual.architecture != model.architecture ||
        !resource_sha256_valid(sha256) || !resource_sha256_valid(actual.sha256) || actual.sha256 != sha256)
        throw std::runtime_error("Native graph model resource identity mismatch");
}

// Called only after authenticated source loading, resource selection and layout validation.
inline void require_native_resource(const Json& graph, const WeightSource& source,
                                     const ResourceIdentity& actual) {
    source.validate();
    const auto& model = source.model();
    require_graph_model(model, graph.value("model_architecture", std::string(models::original.architecture)));
    if (actual.name != model.resource || actual.architecture != model.architecture ||
        !resource_sha256_valid(actual.sha256))
        throw std::runtime_error("Native graph model resource identity mismatch");
    if (graph.contains("reference_model_resource")) {
        const auto& expected = graph.at("reference_model_resource");
        if (expected.at("name").get<std::string>() != model.resource ||
            !resource_sha256_valid(expected.at("sha256").get<std::string>()))
            throw std::runtime_error("Native graph model resource identity mismatch");
    }

    // Disk models are deliberately replaceable compatible weights, authenticated
    // by their own manifest and fixed architecture/layout, not a vendor edition.
    if (source.disk_model())
        return;
    if (graph.contains("reference_model_resource")) {
        const auto& expected = graph.at("reference_model_resource");
        require_resource_identity(model, actual, expected.at("name").get<std::string>(),
                                  expected.at("sha256").get<std::string>());
        return;
    }

    // Migrate old model-free bundles in memory; never rewrite their signed/pinned files.
    const auto container = graph.at("supported_dll_sha256").get<std::string>();
    for (const auto& reference : registry().entries()) {
        if (reference.reference_dll_sha256 == container && reference.architecture == model.architecture &&
            reference.resource == model.resource && !reference.reference_resource_sha256.empty()) {
            require_resource_identity(model, actual, reference.resource, reference.reference_resource_sha256);
            return;
        }
    }
    // Preserve exact-container support for unknown legacy bundles. Without a known
    // resource fingerprint they cannot authorize a different container.
    if (!resource_sha256_valid(container) || source.dll_sha256 != container)
        throw std::runtime_error("Native graph has no verified model resource identity for this DLL");
}
} // namespace phoenix_model
