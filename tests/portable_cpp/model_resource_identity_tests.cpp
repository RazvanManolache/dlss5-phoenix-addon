// Resource-equivalence checks and optional real-bundle loading. Never executes GPU code.
#include "../../native_runtime/model.h"
#include "../../src/native/model/bundle_loader.h"
#include <iostream>

using namespace phoenix_model;
static unsigned checks = 0;
static void check(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
    ++checks;
}
template <class F> static void rejects(F work, std::string_view reason) {
    try { work(); }
    catch (const std::exception& error) {
        check(std::string_view(error.what()).find(reason) != std::string_view::npos, "Unexpected identity rejection");
        return;
    }
    throw std::runtime_error("Invalid resource identity accepted");
}
static Json legacy_graph() {
    return {{"supported_dll_sha256", models::original.reference_dll_sha256}};
}
static Json pinned_graph() {
    return {{"model_architecture", models::original.architecture},
        {"reference_model_resource", {{"name", models::original.resource},
                                      {"sha256", models::original.reference_resource_sha256}}}};
}

static void contracts() {
    const ResourceIdentity identity{std::string(models::original.resource), std::string(models::original.architecture),
                                    std::string(models::original.reference_resource_sha256)};
    WeightSource source{std::filesystem::absolute("not-opened.dll"), std::string(64, '1')};
    for (const auto& graph : {legacy_graph(), pinned_graph()}) {
        require_native_resource(graph, source, identity); ++checks;
        auto wrong = identity; wrong.sha256[0] = wrong.sha256[0] == '0' ? '1' : '0';
        rejects([&] { require_native_resource(graph, source, wrong); }, "resource identity mismatch");
        wrong = identity; wrong.name = "WEIGHTS_CC_SMALL_FP8";
        rejects([&] { require_native_resource(graph, source, wrong); }, "resource identity mismatch");
        wrong = identity; wrong.architecture = "another-network/1";
        rejects([&] { require_native_resource(graph, source, wrong); }, "resource identity mismatch");
        wrong = identity; wrong.sha256 = "";
        rejects([&] { require_native_resource(graph, source, wrong); }, "resource identity mismatch");
    }
    auto graph = pinned_graph();
    graph["reference_model_resource"]["sha256"] = std::string(64, 'X');
    rejects([&] { require_native_resource(graph, source, identity); }, "resource identity mismatch");
    graph = pinned_graph(); graph["reference_model_resource"]["name"] = "wrong";
    rejects([&] { require_native_resource(graph, source, identity); }, "resource identity mismatch");
    graph = legacy_graph(); graph["model_architecture"] = "another-network/1";
    rejects([&] { require_native_resource(graph, source, identity); }, "graph architecture");
    graph = {{"supported_dll_sha256", std::string(64, '2')}};
    rejects([&] { require_native_resource(graph, source, identity); }, "no verified model resource identity");
    graph["supported_dll_sha256"] = source.dll_sha256;
    require_native_resource(graph, source, identity); ++checks; // Existing exact-container legacy contract.
    auto unsupported = source; unsupported.model_id = "small-fp8";
    rejects([&] { require_native_resource(pinned_graph(), unsupported, identity); }, "graph architecture");
    WeightSource folder;
    folder.folder = std::filesystem::absolute("not-opened-folder");
    auto edition = identity; edition.sha256 = std::string(64, '3');
    require_native_resource(pinned_graph(), folder, edition); ++checks; // Explicit compatible disk editions remain allowed.
}

static Json check_dll(const std::filesystem::path& path, const std::filesystem::path& bundle) {
    const auto digest = phoenix_cpp::file_sha(path);
    WeightSource source{path, digest};
    Json result{{"dll_sha256", digest}};
    {
        auto packed = packed_model(source, phoenix_cpp::file_sha);
        check(packed.resource.sha256 == models::original.reference_resource_sha256,
              "Unexpected original-model resource identity");
        require_native_resource(legacy_graph(), source, packed.resource); ++checks;
        require_native_resource(pinned_graph(), source, packed.resource); ++checks;
        result["resource_sha256"] = packed.resource.sha256;
        result["packed_weights_sha256"] = packed.catalogue.at("sha256");
    }
    auto wrong_checksum = source; wrong_checksum.dll_sha256 = std::string(64, '0');
    rejects([&] { packed_model(wrong_checksum, phoenix_cpp::file_sha); }, "SHA-256 mismatch");
    // Alter bytes in memory, preserving the DLL, layout and names. Resource identity must catch it.
    with_dll_resource(source, phoenix_cpp::file_sha, [&](const uint8_t* data, size_t size) {
        std::vector<uint8_t> changed(data, data + size);
        const auto records = resource_records(changed.data(), changed.size());
        changed[records.begin()->second.byte_offset] ^= 1;
        validate_model_layout(changed.data(), changed.size());
        const auto actual = resource_identity(changed.data(), changed.size(), source.model());
        rejects([&] { require_native_resource(legacy_graph(), source, actual); }, "resource identity mismatch");
        rejects([&] { require_native_resource(pinned_graph(), source, actual); }, "resource identity mismatch");
        return 0;
    });
    if (!bundle.empty()) {
        phoenix_bundle::Bundle native(bundle, phoenix_cpp::file_sha, source);
        check(native.generated && native.model_resource.sha256 == models::original.reference_resource_sha256,
              "Native bundle rejected verified model equivalence");
        check(native.weights.at("sha256") == result.at("packed_weights_sha256"), "Native packed weights differ");
        result["native_bundle_loaded"] = true;
        result["kernels_verified"] = native.kernels.size();
    }
    check(phoenix_cpp::file_sha(path) == digest, "Source DLL was changed");
    return result;
}

int main(int argc, char** argv) {
    try {
        contracts();
        std::filesystem::path bundle;
        Json files = Json::array();
        for (int i = 1; i < argc; ++i) {
            if (std::string_view(argv[i]) == "--bundle") {
                if (++i == argc) throw std::runtime_error("Missing bundle argument");
                bundle = std::filesystem::u8path(argv[i]);
            } else files.push_back(check_dll(std::filesystem::u8path(argv[i]), bundle));
        }
        std::cout << Json({{"passed", true}, {"cpu_only", true}, {"checks", checks}, {"external_models", files}}).dump() << '\n';
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
