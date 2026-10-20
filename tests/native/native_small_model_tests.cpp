// CPU-only real-bundle/model loading and injected NVAPI recording. No GPU calls.
#include "../../src/native/support/dispatch.h"
#include <iostream>
#include <set>

using Json = nlohmann::json;
static unsigned checks = 0;
static void verify(bool value, const char* error) {
    ++checks;
    if (!value) throw std::runtime_error(error);
}
static std::vector<std::vector<uint8_t>> recorded;
static NvAPI_Status __cdecl capture(ID3D12GraphicsCommandList*, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* launches, NvU32 count) {
    for (unsigned index = 0; index < count; ++index) {
        const auto* data = static_cast<const uint8_t*>(launches[index].pParams);
        recorded.emplace_back(data, data + launches[index].paramSize);
    }
    return NVAPI_OK;
}
static std::string digest(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read authenticated model/bundle");
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    return phoenix_model::weight_digest(bytes.data(), bytes.size());
}
template<class F> static void rejects(F function) {
    bool failed = false;
    try { function(); } catch (const std::exception&) { failed = true; }
    verify(failed, "Invalid native model/plan accepted");
}
static void storage(const Json& frame) {
    const auto extent = frame.at("arenas").at("workspace").at("bytes").get<uint64_t>();
    const auto& allocation = frame.at("allocation");
    for (const auto& a : allocation) {
        const auto begin = a.at("offset").get<uint64_t>(), end = begin + a.at("bytes").get<uint64_t>();
        verify(end <= extent && begin % 512 == 0, "Workspace allocation outside arena");
        for (const auto& b : allocation) {
            if (a.at("tensor") == b.at("tensor") || a.at("last_group").get<unsigned>() < b.at("first_group").get<unsigned>() ||
                b.at("last_group").get<unsigned>() < a.at("first_group").get<unsigned>()) continue;
            const auto other = b.at("offset").get<uint64_t>();
            verify(end <= other || other + b.at("bytes").get<uint64_t>() <= begin, "Live native tensors alias");
        }
    }
}
static void repeat_checks(const Json& frame) {
    const auto repeat = phoenix_addon::native_repeat_plan(frame);
    verify(repeat.stride > 0, "Repeat counter extent missing");
    const std::set<unsigned> expected[]{{56},{32,48,56},{40,56,64},{40,48},{32,48,56}};
    for (unsigned block = 0; block < 8; ++block)
        for (unsigned stage = 0; stage < 5; ++stage) {
            std::set<unsigned> actual;
            for (const auto& field : repeat.fields[block][stage]) {
                verify(field.second < repeat.stride && field.second % 4 == 0, "Repeat pointer outside counter bank");
                actual.insert(field.first);
            }
            auto required = expected[stage];
            if (block == 7 && stage == 4) required.erase(56); // final projection waits, without publishing
            verify(actual == required, "Repeat synchronization patched wrong stage/ABI slot");
        }
}
static void dispatch_checks(const Json& frame) {
    using namespace phoenix_addon;
    std::vector<std::vector<uint8_t>> arguments;
    std::vector<NVAPI_CU_KERNEL_LAUNCH_PARAMS> launches;
    for (unsigned index = 1; index <= 78; ++index) {
        const auto plan = phoenix_bundle::launch(frame.at("launches").at(index), [](const Json&) { return 42ull; });
        arguments.push_back(phoenix_launch_plan::materialize(plan,
            {{{0x1000000000ull, frame.at("arenas").at("workspace").at("bytes")},
              {0x8000000000ull, frame.at("arenas").at("weights").at("bytes")}}}));
        NVAPI_CU_KERNEL_LAUNCH_PARAMS launch{};
        launch.paramSize = unsigned(arguments.back().size());
        launches.push_back(launch);
    }
    for (unsigned index = 0; index < 78; ++index) launches[index].pParams = arguments[index].data();
    const auto before = arguments;
    auto* command = reinterpret_cast<ID3D12GraphicsCommandList*>(uintptr_t(0x1234));
    Controls requested;
    requested.tone = 0.375f; requested.structure = 1.5f; requested.reset = 1;
    const auto resolved = resolve_controls(requested, frame.at("width"), frame.at("height"), true, true, 7);
    unsigned cases = 0;
    for (unsigned depth = 1; depth <= 8; ++depth)
        for (bool batch : {false, true}) {
            recorded.clear();
            FrameDispatch wrapper(capture);
            wrapper.configure_repeats(nullptr,frame,{1,1,1,1,1,1,1,1});
            FrameDispatch::Scope scope(wrapper, command, resolved, depth);
            if (batch) verify(FrameDispatch::dispatch_function()(command, launches.data(), 78) == NVAPI_OK, "Batched dispatch");
            else for (const auto& launch : launches)
                verify(FrameDispatch::dispatch_function()(command, &launch, 1) == NVAPI_OK, "Individual dispatch");
            scope.finish();
            verify(recorded.size() == 38 + 5 * depth, "Small depth dispatch launch count");
            auto stem = arguments.front(), head = arguments.back();
            patch_stem(stem.data(), stem.size(), resolved); patch_head(head.data(), head.size(), resolved);
            verify(recorded.front() == stem && recorded.back() == head, "Controls patched wrong native boundary");
            unsigned output = 1;
            for (unsigned input = 1; input < 77; ++input) {
                if (input >= 19 + 5 * depth && input < 59) continue;
                auto expected = arguments[input];
                if (depth < 8 && input == 59)
                    memcpy(expected.data(), arguments[18 + 5 * depth].data() + 16, 8);
                verify(recorded.at(output++) == expected, "Interior dispatch or depth reconnection changed");
            }
            verify(arguments == before, "Dispatch modified immutable plan arguments");
            ++cases;
        }
    for (unsigned group = 0; group < 15; ++group) {
        recorded.clear(); FrameDispatch wrapper(capture); wrapper.configure_plan(frame);
        skip_groups_mask = 1u << group;
        try {
            FrameDispatch::Scope scope(wrapper, command, resolved);
            for (const auto& launch : launches) FrameDispatch::dispatch_function()(command, &launch, 1);
            scope.finish();
        } catch (...) { skip_groups_mask = 0; throw; }
        skip_groups_mask = 0;
        const unsigned sizes[]{1,2,2,2,2,9,1,40,1,9,2,2,2,2,1};
        verify(recorded.size() == 78 - sizes[group], "Small skip group selected wrong launches");
    }
}
int main(int argc, char** argv) {
    try {
        verify(argc == 4, "Expected project, bundle and 310.9.16 DLL paths");
        verify(!GetModuleHandleW(L"nvapi64.dll") && !GetModuleHandleW(L"nvngx_dlssnr.dll"), "GPU/vendor library already loaded");
        const std::filesystem::path root = argv[1], folder = argv[2], dll = argv[3];
        const auto original = phoenix_bundle::json_file(root / "src/native/model/production_native_graph.json");
        const auto snapshot = original.dump();
        phoenix_model::WeightSource source = phoenix_model::WeightSource::config({{"kind", "nvidia_dll"},
            {"path", dll.string()}, {"sha256", digest(dll)}, {"model", "small-fp8"}});
        phoenix_bundle::Bundle bundle(folder, digest, source);
        verify(bundle.graph == phoenix_model::models::small_native_graph(original), "Bundle selected different small graph");
        verify(original.dump() == snapshot, "Original graph was mutated");
        verify(bundle.weights.at("records").size() == 77 && bundle.kernels.size() == 41, "Small native resource/kernel count");
        verify(bundle.model_resource.name == "WEIGHTS_CC_SMALL_FP8" &&
            bundle.model_resource.sha256 == "f743e8eb320f42bc9bdc992ee2bf51f7bb7c8688ced4ecf5ce6807ec4eef7962",
            "Native bundle selected wrong model resource");
        const auto specs = phoenix_native::catalog(bundle);
        const auto packed = bundle.load_weights();
        verify(packed.size() == bundle.weights.at("bytes"), "Packed small resource extent");
        verify(bundle.embedded_weights.empty(), "Bundle retained consumed CPU weight storage");
        verify(bundle.load_weights() == packed, "Session recreation reloaded different weights");
        std::set<std::string> records;
        for (const auto& item : bundle.weights.at("records")) records.insert("weight:" + item.at("name").get<std::string>());
        std::set<std::string> used;
        std::map<std::string, unsigned> publications;
        for (const auto& layer : bundle.graph.at("layers")) {
            const auto index = layer.at("index").get<unsigned>();
            for (const auto& binding : layer.at("bindings").items()) {
                const auto& t = binding.value().at("target"); if (!t.is_string()) continue;
                const auto name = t.get<std::string>(), role = binding.value().at("role").get<std::string>();
                if (name.rfind("weight:", 0) == 0) { verify(records.count(name), "Native weight binding missing"); used.insert(name); }
                if (role.rfind("publication", 0) == 0) publications[name] = index;
                if (role.rfind("wait", 0) == 0) verify(publications.count(name) && publications.at(name) < index, "Wait points at removed/unproduced publication");
            }
        }
        verify(used == records, "Some small model weights were omitted");
        const auto& layers = bundle.graph.at("layers");
        // Independent architecture checkpoints: skip pairs, history scalar, and
        // shifted attention grids differ from merely shortening the old graph.
        verify(layers.at(60).at("bindings").at("8").at("target") == "tensor:055.output_encoder_skip", "512 skip");
        const unsigned decoder_entries[]{69,71,73,75};
        const char* expected_skips[]{"tensor:023.output","tensor:015.output","tensor:009.output","tensor:005.output"};
        for (unsigned stage = 0; stage < 4; ++stage) {
            const auto& first = layers.at(decoder_entries[stage]); const auto& second = layers.at(decoder_entries[stage]+1);
            const bool small_channels = stage == 3;
            verify(first.at("bindings").at(small_channels ? "80" : "24").at("target") == expected_skips[stage], "Encoder/decoder skip mismatch");
            verify(first.at("scalars").at(small_channels ? "32" : "40") == uint32_t(-4) &&
                first.at("scalars").at(small_channels ? "36" : "44") == 0, "Decoder phase two");
            verify(second.at("scalars").at(small_channels ? "32" : "40") == 0 &&
                second.at("scalars").at(small_channels ? "36" : "44") == uint32_t(-4), "Decoder phase three");
            verify(first.at("grid").at(0).get<std::string>().find("+4)") != std::string::npos, "Shifted decoder grid missing margin");
        }
        verify(layers.back().at("bindings").at("104").at("target") == "weight:block30.layer0.blend_scale", "Wrong model history gain");
        verify(layers.at(15).at("grid").at(0) == "ceil8(W5+4)*1" &&
            layers.at(63).at("grid").at(0) == "ceil8(W5+4)*1", "Split attention phase grid");
        Json resolutions = Json::array();
        for (const auto size : std::vector<std::array<unsigned,2>>{{1,1},{16,12},{320,320},{384,320},{640,360},
                {801,603},{960,540},{1024,768},{1920,1080},{2560,1440},{3840,2160},{1080,1920}}) {
            const auto& frame = bundle.select_frame(size[0],size[1], [&](const Json& f) { phoenix_native::preflight(f,specs); });
            const auto info = phoenix_native::preflight(frame,specs);
            verify(info.neural_launches == 78 && info.group_ends.size() == 15 && info.group_ends.back() == 78, "Small plan schedule");
            storage(frame);
            repeat_checks(frame);
            if (size[0] == 384) dispatch_checks(frame);
            resolutions.push_back({{"width",size[0]},{"height",size[1]}, {"workspace_bytes",frame.at("arenas").at("workspace").at("bytes")}});
        }
        const auto frame = phoenix_generated::frame(bundle.graph,bundle.kernels,bundle.weights,640,360);
        for (unsigned mutation = 0; mutation < 5; ++mutation) {
            auto bad = frame;
            if (mutation == 0) bad.erase("model_architecture");
            if (mutation == 1) bad["launches"][3]["name"] = "cc_vit_1d_qkv_chained_fp8";
            if (mutation == 2) bad["neural_region_barriers"][3]["after_launch"] = 10;
            if (mutation == 3) bad["launches"][78]["relocations"][0]["relative_offset"] = 0;
            if (mutation == 4) bad["model_architecture"] = "unknown";
            rejects([&] { phoenix_native::preflight(bad,specs); });
        }
        // The same authenticated bundle continues to select the original model
        // from this DLL. Its generated launch contract is unchanged.
        auto original_source = source;
        original_source.model_id = "original";
        phoenix_bundle::Bundle original_bundle(folder,digest,original_source);
        const auto original_bundle_snapshot = phoenix_bundle::json_file(folder /
            (original_bundle.manifest.contains("native_graph_v2_sha256") ? "native_graph_v2.json" : "native_graph.json"));
        verify(original_bundle.graph == original_bundle_snapshot && original_bundle.weights.at("records").size() == 153,
            "Original native bundle selection changed");
        for (const auto size : std::vector<std::array<unsigned,2>>{{320,320},{801,603},{1920,1080}}) {
            const auto& original_frame = original_bundle.select_frame(size[0],size[1], [&](const Json& f) { phoenix_native::preflight(f,specs); });
            verify(phoenix_native::preflight(original_frame,specs).neural_launches == 154, "Original native schedule changed");
            repeat_checks(original_frame);
            verify(original_frame == phoenix_generated::frame(original,original_bundle.kernels,original_bundle.weights,size[0],size[1]),
                "Original launch bytes differ between legacy and current source graph");
        }
        auto obsolete = original; obsolete["layers"].erase(obsolete["layers"].begin());
        rejects([&] { phoenix_model::models::small_native_graph(obsolete); });
        verify(!GetModuleHandleW(L"nvapi64.dll") && !GetModuleHandleW(L"nvngx_dlssnr.dll"), "Test loaded GPU/vendor code");
        std::cout << Json({{"passed",true},{"checks",checks},{"model","small-fp8"},{"weight_records",77},
            {"neural_launches",78},{"shared_kernel_modules",41},{"dispatch_depth_cases",16},{"skip_group_cases",15},
            {"gpu_used",false},{"vendor_reference_used",false},{"resolutions",resolutions}}).dump() << '\n';
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
