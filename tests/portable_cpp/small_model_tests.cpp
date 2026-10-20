// Real-resource decoder and graph contracts. No GPU is created or executed.
#include "../../native_runtime/graph.h"
#include "../../src/integration/compatibility/session_policy.h"
#include <iostream>
#include <set>
using namespace phoenix_cpp;

class RecordingBackend final : public Backend {
    uintptr_t next_ = 0x100000;
  public:
    uint64_t live = 0;
    const char* name() const override { return "recording-only"; }
    Allocation allocate(uint64_t bytes) override {
        const auto address = next_;
        next_ += uintptr_t((bytes + 255) & ~uint64_t(255));
        live += bytes;
        return {reinterpret_cast<void*>(address), bytes, address};
    }
    void release(Allocation value) override { live -= value.bytes; }
    void upload(Allocation value, const void*, uint64_t bytes) override { demand(bytes <= value.bytes, "Upload bounds"); }
    void download(Allocation, void*, uint64_t) override { throw std::runtime_error("Recording backend cannot execute"); }
    void execute(const std::vector<Command>&, std::vector<Event>&) override { throw std::runtime_error("Recording backend cannot execute"); }
    void wait() override {}
};

int main(int argc, char** argv) {
    try {
        demand(argc == 2, "Usage: small_model_tests NEW_DLL");
        const auto path = std::filesystem::absolute(std::filesystem::u8path(argv[1]));
        const auto before = file_sha(path);
        phoenix_model::WeightSource source{path, before};
        source.model_id = "small-fp8";
        Model model(source);
        std::string catalog;
        unsigned tensors = 0;
        model.visit_tensors([&](const auto& name, const auto& bytes) {
            ++tensors;
            catalog += name + '\t' + std::to_string(bytes.size()) + '\t' +
                       phoenix_model::weight_digest(bytes.data(), bytes.size()) + '\n';
        });
        const auto digest = phoenix_model::weight_digest(reinterpret_cast<const uint8_t*>(catalog.data()), catalog.size());
        demand(tensors == 302 && digest == "cb1269466e5494f0aae678f3eb7801655efde2e065d1febe6e2ec5c3f638dd3f",
               "Smaller model tensor decoding differs from the lossless extraction");
        demand(model.history_gain() == .76708984375f, "Smaller model history gain");
        Json plans = Json::array();
        for (const auto extent : {320u, 384u}) {
            RecordingBackend backend;
            {
                Graph graph(backend, model, false, 128);
                graph.build(extent, 320, {1,1,1,1,1,1,1,1});
                demand(graph.output.channels == 4 && graph.output.width == extent && graph.output.height == 320,
                       "Smaller graph output extent");
                std::set<unsigned> blocks;
                for (const auto& command : graph.commands) {
                    demand(command.block <= 30, "Original block leaked into smaller graph");
                    blocks.insert(command.block);
                    const auto& op = command.operation;
                    const unsigned kind = op.flags & 255;
                    if (op.family == phx::Elementwise && kind == phx::GatherWindowBatch &&
                        command.block >= 20 && command.block <= 29) {
                        const unsigned index = (command.block >= 22) ? (command.block - 22) % 2 : command.block - 20;
                        const unsigned expected = index == 0 ? 4u : 4u << 16;
                        demand(op.reduction_count == expected, "Smaller decoder shift phase must be 2,3 at every level");
                    }
                }
                demand(blocks.size() == 31, "Incomplete smaller graph");
                for (unsigned index = 0; index <= 30; ++index)
                    demand(blocks.contains(index), "Missing smaller block");
                plans.push_back({{"width", extent}, {"height",320}, {"blocks",blocks.size()},
                                 {"commands",graph.commands.size()}, {"weight_bytes",graph.weight_bytes()},
                                 {"resident_bytes",graph.memory_bytes()}});
            }
            demand(backend.live == 0, "Graph allocations leaked");
        }
        phoenix_compat::Settings settings;
        const auto initial = settings;
        settings.model_id = "small-fp8";
        demand(settings.valid() && phoenix_compat::helper_settings_changed(initial, settings),
               "UI/helper model selection does not recreate its session");
        demand(phoenix_compat::settings_json(settings).at("ModelId") == "small-fp8", "Helper lost model identity");
        settings.model_id = "unknown";
        demand(!settings.valid(), "Unknown UI model accepted");
        demand(file_sha(path) == before, "Supplied DLL changed");
        std::cout << Json({{"passed",true},{"gpu_used",false},{"graph_executed",false},
            {"decoded_tensors",tensors},{"tensor_catalogue_sha256",digest},
            {"history_gain",model.history_gain()},{"plans",plans}}).dump(2) << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
