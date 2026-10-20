// Records complete portable/native plans without creating or executing a GPU.
// Compare this probe's JSON before/after a host-side graph refactor.
#include "../../native_runtime/graph.h"
#include "../../src/native/model/models/small_fp8/native_graph.h"
#include <iostream>

using namespace phoenix_cpp;

static std::string fingerprint(const Json& value) {
    const auto bytes = value.dump();
    return phoenix_model::weight_digest(reinterpret_cast<const uint8_t*>(bytes.data()),
                                        bytes.size());
}

class RecordingBackend final : public Backend {
    uint64_t next_address_ = 0x100000;

  public:
    uint64_t live_bytes = 0;
    Json uploads = Json::array();
    const char* name() const override {
        return "recording-only";
    }
    Allocation allocate(uint64_t bytes) override {
        const auto address = next_address_;
        next_address_ += (bytes + 255) & ~uint64_t(255);
        live_bytes += bytes;
        return {phx::address_from_bits(address), bytes, address};
    }
    void release(Allocation allocation) override {
        live_bytes -= allocation.bytes;
    }
    void upload(Allocation allocation, const void* bytes, uint64_t count) override {
        demand(count <= allocation.bytes, "Upload exceeds allocation");
        uploads.push_back(
            {allocation.token, count,
             phoenix_model::weight_digest(static_cast<const uint8_t*>(bytes), size_t(count))});
    }
    void download(Allocation, void*, uint64_t) override {
        throw std::runtime_error("Recording only");
    }
    void execute(const std::vector<Command>&, std::vector<Event>&) override {
        throw std::runtime_error("Recording only");
    }
    void wait() override {}
};

static Json commands(const Graph& graph) {
    Json result = Json::array();
    for (const auto& command : graph.commands) {
        const auto& op = command.operation;
        // Serialize fields explicitly: object padding is not part of the ABI.
        result.push_back({command.block,
                          command.layer,
                          command.count,
                          phx::address_bits(op.left_input),
                          phx::address_bits(op.right_input),
                          phx::address_bits(op.output),
                          phx::address_bits(op.auxiliary),
                          phx::address_bits(op.secondary_output),
                          op.family,
                          op.flags,
                          op.row_count,
                          op.column_count,
                          op.reduction_count,
                          op.batch_count,
                          op.left_batch_stride,
                          op.right_batch_stride,
                          op.output_batch_stride,
                          op.left_row_stride,
                          op.right_row_stride,
                          op.output_row_stride,
                          std::bit_cast<uint32_t>(op.scale)});
    }
    return result;
}

int main(int argc, char** argv) {
    try {
        demand(argc == 3, "Expected project root and model DLL");
        const auto root = std::filesystem::u8path(argv[1]);
        std::ifstream stream(root / "src/native/model/production_native_graph.json");
        Json original;
        stream >> original;
        const auto original_snapshot = fingerprint(original);
        const auto native = phoenix_model::models::small_native_graph(original);
        demand(fingerprint(original) == original_snapshot, "Original graph mutated");
        Json result{{"native_graph_sha256", fingerprint(native)},
                    {"original_native_graph_sha256", original_snapshot},
                    {"gpu_used", false},
                    {"inference_executed", false},
                    {"portable", Json::array()}};
        const auto dll = std::filesystem::absolute(std::filesystem::u8path(argv[2]));
        phoenix_model::WeightSource source{dll, file_sha(dll)};
        for (const auto* model_id : {"original", "small-fp8"}) {
            source.model_id = model_id;
            Model model(source);
            for (unsigned configuration = 0; configuration < 4; ++configuration) {
                const bool fused = configuration == 2;
                const bool device_features = configuration == 3;
                const unsigned width = configuration == 1 ? 801 : 16;
                const unsigned height = configuration == 1 ? 603 : 12;
                const std::array<unsigned, 8> repeats =
                    configuration == 1 ? std::array<unsigned, 8>{0, 2, 1, 0, 1, 1, 3, 1}
                                       : std::array<unsigned, 8>{1, 1, 1, 1, 1, 1, 1, 1};
                RecordingBackend backend;
                {
                    Graph graph(backend, model, fused, 128, device_features, fused,
                                device_features);
                    graph.build(width, height, repeats);
                    result["portable"].push_back(
                        {{"model", model_id},
                         {"configuration", configuration},
                         {"width", width},
                         {"height", height},
                         {"repeats", repeats},
                         {"commands", graph.commands.size()},
                         {"command_sha256", fingerprint(commands(graph))},
                         {"uploads_sha256", fingerprint(backend.uploads)},
                         {"resident_bytes", graph.memory_bytes()},
                         {"weight_bytes", graph.weight_bytes()},
                         {"input_address", phx::address_bits(graph.input.ptr())},
                         {"output_address", phx::address_bits(graph.output.ptr())}});
                }
                demand(backend.live_bytes == 0, "Recording allocations leaked");
            }
        }
        std::ifstream reference_stream(root / "tests/fixtures/small_model_graph_fingerprints.json");
        demand(bool(reference_stream), "Missing pre-refactor graph fingerprints");
        Json reference;
        reference_stream >> reference;
        demand(result == reference, "Model commands, weight uploads or native graph changed");
        result["passed"] = true;
        result["pre_refactor_baseline_verified"] = true;
        std::cout << result.dump(2) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
