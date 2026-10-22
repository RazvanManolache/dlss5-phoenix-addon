// CPU-only contracts: model identity, dispatch, parsing, and legacy compatibility.
#include "../../native_runtime/graph.h"
#include "../../src/native/model/bundle_loader.h"
#include <iostream>

using namespace phoenix_model;
static unsigned checks = 0;
static void check(bool result, const char* message) {
    if (!result) throw std::runtime_error(message);
    ++checks;
}
template <class F> static void rejects(F operation, std::string_view message) {
    try { operation(); }
    catch (const std::exception& error) {
        check(std::string_view(error.what()).find(message) != std::string_view::npos,
              "Unexpected model rejection reason");
        return;
    }
    throw std::runtime_error("Invalid model selection was accepted");
}

static std::vector<uint8_t> synthetic_resource(unsigned records) {
    std::vector<uint8_t> data(8);
    auto integer = [&](uint64_t value) {
        const auto offset = data.size(); data.resize(offset + 8);
        std::memcpy(data.data() + offset, &value, 8);
    };
    for (unsigned i = 0; i < records; ++i) {
        const auto name = "block" + std::to_string(i) + ".layer0.layer";
        integer(name.size()); data.insert(data.end(), name.begin(), name.end());
        integer(21); integer(21); integer(1);
        data.insert(data.end(), 4, 0); data.push_back(uint8_t(i));
    }
    const uint64_t size = data.size(); std::memcpy(data.data(), &size, 8);
    return data;
}

static void record_format_edges(const std::vector<uint8_t>& valid) {
    const auto records = resource_records(valid.data(), valid.size());
    const auto payload = records.at("block0.layer0.layer").byte_offset;
    const auto header = payload - resource_format::payload_offset;
    auto corrupt = [&](size_t offset, auto value, std::string_view reason) {
        auto bytes = valid;
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
        rejects([&] { resource_records(bytes.data(), bytes.size()); }, reason);
    };
    corrupt(8, uint64_t(0), "name extent");
    corrupt(8, uint64_t(4097), "name extent");
    corrupt(header - 8, uint64_t(19), "body extent");
    corrupt(header, uint64_t(20), "Invalid DLL weight record");
    corrupt(header + 8, uint64_t(0), "Invalid DLL weight record");
    corrupt(header + 8, uint64_t(2), "Invalid DLL weight record");
    corrupt(header + 16, uint32_t(2), "Invalid DLL weight record");
    auto device_one = valid;
    const uint32_t device = 1;
    std::memcpy(device_one.data() + header + 16, &device, sizeof(device));
    check(resource_records(device_one.data(), device_one.size()).size() == records.size(),
          "Device tag one must remain supported");
    auto truncated = valid;
    truncated.pop_back();
    const uint64_t truncated_size = truncated.size();
    std::memcpy(truncated.data(), &truncated_size, sizeof(truncated_size));
    rejects([&] { resource_records(truncated.data(), truncated.size()); }, "body extent");
}

static void resource_selection() {
    std::vector<uint8_t> pe(2048);
    auto word = [&](size_t offset, uint32_t value) {
        pe.at(offset) = uint8_t(value); pe.at(offset + 1) = uint8_t(value >> 8);
    };
    auto dword = [&](size_t offset, uint32_t value) {
        word(offset, value); word(offset + 2, value >> 16);
    };
    word(0, 0x5a4d); dword(0x3c, 64); dword(64, 0x4550);
    word(70, 1); word(84, 240); word(88, 0x20b); dword(88 + 108, 16);
    dword(88 + 128, 0x1000); dword(88 + 132, 1024);
    dword(328 + 12, 0x1000); dword(328 + 16, 1536); dword(328 + 20, 512);
    constexpr size_t base = 512;
    word(base + 14, 1); dword(base + 16, 10); dword(base + 20, 0x80000020);
    word(base + 0x20 + 12, 2);
    dword(base + 0x30, 0x80000100); dword(base + 0x34, 0x80000080);
    dword(base + 0x38, 0x80000140); dword(base + 0x3c, 0x800000a0);
    for (const auto& entry : {std::pair<size_t, std::string_view>{0x100, models::original.resource},
                              {0x140, models::small_fp8.resource}}) {
        word(base + entry.first, unsigned(entry.second.size()));
        for (size_t i = 0; i < entry.second.size(); ++i)
            word(base + entry.first + 2 + 2 * i, entry.second[i]);
    }
    for (size_t table : {0x80, 0xa0}) {
        word(base + table + 14, 1); dword(base + table + 16, 1033);
        dword(base + table + 20, table == 0x80 ? 0xc0 : 0xd0);
    }
    dword(base + 0xc0, 0x1300); dword(base + 0xc4, 4);
    dword(base + 0xd0, 0x1310); dword(base + 0xd4, 4);
    pe[base + 0x300] = 17; pe[base + 0x310] = 29;
    check(pe_weight_resource(pe).front() == 17, "Legacy PE selector changed");
    size_t inspected=0;
    const auto located=pe_weight_resource_range(pe.size(),[&](size_t offset,void* data,size_t count){
        check(offset<=pe.size() && count<=pe.size()-offset,"PE reader exceeded file bounds");
        check(offset<base+0x300,"Locating a resource must not read its payload");
        inspected+=count; std::memcpy(data,pe.data()+offset,count);
    });
    check(located==std::pair<size_t,size_t>{base+0x300,4} && inspected<pe.size(),
          "Bounded PE resource location differs from the span parser");
    auto truncated=pe; truncated.resize(70);
    rejects([&]{pe_weight_resource(truncated);},"Truncated PE model file");

    check(pe_weight_resource(pe, models::small_fp8.resource).front() == 29,
          "PE selector returned the wrong model");
    rejects([&] { pe_weight_resource(pe, "missing"); }, "selected model resource");
    rejects([&] { pe_weight_resource(pe, ""); }, "Invalid model resource name");
    word(base + 0x102, 'X');
    rejects([&] { pe_weight_resource(pe); }, "selected model resource");
    check(pe_weight_resource(pe, models::small_fp8.resource).front() == 29,
          "Missing original model affected explicit new resource selection");
}

static void selection_contracts() {
    const auto& original = definition();
    check(original.id == "original" && original.implemented, "Original default changed");
    check(original.record_count == 153 && original.decoded_tensor_count == 658,
          "Original model contract changed");
    const auto& lightweight = definition("small-fp8");
    check(lightweight.implemented && lightweight.record_count == 77 && lightweight.decoded_tensor_count == 302,
          "Lightweight model contract changed");
    phoenix_cpp::Model::require_supported(lightweight);
    phoenix_cpp::Graph::require_supported(lightweight);
    rejects([] { definition("unknown"); }, "Unknown Phoenix DLSS5 model");
    const Json legacy = {{"kind", "nvidia_dll"}, {"path", "not-opened.dll"}, {"sha256", std::string(64, '0')}};
    auto source = WeightSource::config(legacy);
    check(source.model().id == "original" && source.configuration() == legacy,
          "Legacy source roundtrip changed");
    auto explicit_original = legacy; explicit_original["model"] = "original";
    check(WeightSource::config(explicit_original).configuration() == legacy,
          "Explicit original is not the legacy default");
    auto selected = legacy; selected["model"] = "small-fp8";
    source = WeightSource::config(selected);
    check(source.configuration() == selected, "Selected model lost during serialization");
    auto files = WeightSource::config({{"kind", "files"}, {"model", "small-fp8"}});
    check(files.configuration().at("model") == "small-fp8", "File source dropped selected model");
    rejects([] { WeightSource::config({{"model", ""}}); }, "Unknown Phoenix DLSS5 model");
    rejects([] { WeightSource::config({{"model", "unknown"}}); }, "Unknown Phoenix DLSS5 model");
    auto unavailable = lightweight;
    unavailable.implemented = false;
    unavailable.unavailable_reason = "Fixture architecture unavailable";
    rejects([&] { phoenix_cpp::Model::require_supported(unavailable); }, unavailable.unavailable_reason);
    rejects([&] { phoenix_cpp::Graph::require_supported(unavailable); }, unavailable.unavailable_reason);
    auto folder = WeightSource::config({{"kind", "model_folder"},
        {"path", source_path_string(std::filesystem::absolute("not-opened"))}, {"model", "small-fp8"}});
    check(folder.configuration().at("model") == "small-fp8", "Folder source lost selection");
    require_graph_model(original, "phoenix-network37/1");
    rejects([&] { require_graph_model(original, lightweight.architecture); }, "graph architecture");
    rejects([&] { require_graph_model(lightweight, original.architecture); }, "graph architecture");
    auto records = synthetic_resource(153);
    check(resource_records(records.data(), records.size()).size() == 153, "Original parser changed");
    record_format_edges(records);
    rejects([&] { resource_records(records.data(), records.size(), lightweight); }, "record count");
    auto small_records = synthetic_resource(77);
    check(resource_records(small_records.data(), small_records.size(), lightweight).size() == 77,
          "Resource parser not parameterized by model");
    rejects([&] { resource_records(small_records.data(), small_records.size()); }, "record count");
    rejects([&] { validate_model_layout(records.data(), records.size()); }, "record names or sizes");
    rejects([&] { validate_model_layout(small_records.data(), small_records.size(), lightweight); }, "record names or sizes");
    const auto catalogue = model_catalogue();
    check(catalogue.size() == registry().entries().size(), "Catalogue dropped a registered model");
    for (const auto& model : registry().entries()) {
        const auto entry = std::find_if(catalogue.begin(), catalogue.end(), [&](const Json& item) {
            return item.at("id").get<std::string>() == model.id;
        });
        check(entry != catalogue.end() && entry->at("implemented") == model.implemented &&
              entry->at("architecture").get<std::string>() == model.architecture,
              "Capability catalogue misrepresents model availability");
    }
    resource_selection();
}

static void extensible_registry_contracts() {
    auto third = models::original;
    third.id = "fixture-third";
    third.label = "Fixture third edition";
    third.resource = "WEIGHTS_FIXTURE_THIRD";
    const std::array entries{models::original, models::small_fp8, third};
    const Registry three{entries};
    check(three.find("fixture-third").resource == "WEIGHTS_FIXTURE_THIRD", "Third model not resolved");
    check(model_catalogue(three).size() == entries.size(), "Third model missing from catalogue");
    // Different weight editions can share an existing architecture implementation.
    phoenix_cpp::Model::require_supported(three.find("fixture-third"));
    phoenix_cpp::Graph::require_supported(three.find("fixture-third"));
    check(three.find("fixture-third").architecture == models::original.architecture,
          "Registry forced each model to invent a new architecture");
    auto future_architecture = third;
    future_architecture.architecture = "fixture-future-network/1";
    rejects([&] { phoenix_cpp::Model::require_supported(future_architecture); }, "No Phoenix DLSS5 tensor decoder");
    rejects([&] { phoenix_cpp::Graph::require_supported(future_architecture); }, "No Phoenix DLSS5 graph builder");
    // A support flag alone must never route an unknown graph to the original.
    check(future_architecture.implemented, "Future architecture test did not exercise implemented flag");
    std::vector<std::string> names;
    names.reserve(32);
    std::vector<Definition> many;
    many.reserve(32);
    for (unsigned index = 0; index < 32; ++index) {
        names.push_back("fixture-edition-" + std::to_string(index));
        auto entry = models::original;
        entry.id = names.back();
        many.push_back(entry);
    }
    const Registry thirty_two{many};
    const auto catalogue = model_catalogue(thirty_two);
    check(catalogue.size() == many.size(), "Registry has a fixed model-count ceiling");
    for (const auto& entry : many)
        check(thirty_two.find(entry.id).id == entry.id, "Large registry lookup failed");
    rejects([&] { thirty_two.find("missing"); }, "Unknown Phoenix DLSS5 model");
    const std::array duplicates{third, third};
    rejects([&] { Registry invalid{duplicates}; }, "Duplicate Phoenix DLSS5 model ID");
    auto incomplete = third;
    incomplete.id = "";
    const std::array invalid_entries{incomplete};
    rejects([&] { Registry invalid{invalid_entries}; }, "Incomplete Phoenix DLSS5 model definition");
    check(model_catalogue().size() == registry().entries().size(), "Fixture models leaked into production registry");
}

static Json decode_original(const std::filesystem::path& dll) {
    const auto before = phoenix_cpp::file_sha(dll);
    WeightSource source{dll, before};
    phoenix_cpp::Model model(source);
    size_t count = 0;
    std::string catalogue;
    model.visit_tensors([&](const auto& name, const auto& data) {
        ++count;
        catalogue += name + '\t' + std::to_string(data.size()) + '\t' + weight_digest(data.data(), data.size()) + '\n';
    });
    const auto sha = weight_digest(reinterpret_cast<const uint8_t*>(catalogue.data()), catalogue.size());
    check(count == 658 && sha == "db9949368cd15e17886ec5f134e117c1de52889db6db6e4dd59f456bd702743e",
          "Original tensor decoding changed after refactor");
    check(before == phoenix_cpp::file_sha(dll), "External model file changed");
    check(std::isfinite(model.history_gain()), "Original history gain unavailable");
    return {{"dll_sha256", before}, {"model", model.definition().id},
            {"decoded_tensors", count}, {"tensor_catalogue_sha256", sha}};
}

int main(int argc, char** argv) {
    try {
        selection_contracts();
        extensible_registry_contracts();
        Json models = Json::array();
        // Optional explicit reference files are never required by CI/CTest.
        for (int i = 1; i < argc; ++i) models.push_back(decode_original(std::filesystem::u8path(argv[i])));
        std::cout << Json({{"passed", true}, {"cpu_only", true}, {"checks", checks},
                           {"external_models", models}}).dump(2) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
