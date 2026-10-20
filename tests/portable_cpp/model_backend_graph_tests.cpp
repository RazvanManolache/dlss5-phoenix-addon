// Authenticated model graph recording for backend feature profiles; no GPU/device/execute.
#include "../../native_runtime/graph.h"
#include <iostream>
using namespace phoenix_cpp;
class Recorder : public Backend {
    uint64_t address_ = 0x100000;

  public:
    uint64_t live = 0;
    const char* name() const override {
        return "recording-only";
    }
    Allocation allocate(uint64_t size) override {
        const auto address = address_;
        address_ += (size + 255) & ~uint64_t(255);
        live += size;
        return {phx::address_from_bits(address), size, address};
    }
    void release(Allocation allocation) override {
        live -= allocation.bytes;
    }
    void upload(Allocation allocation, const void*, uint64_t size) override {
        demand(size <= allocation.bytes, "Upload bounds");
    }
    void download(Allocation, void*, uint64_t) override {
        throw std::runtime_error("Recording only");
    }
    void execute(const std::vector<Command>&, std::vector<Event>&) override {
        throw std::runtime_error("Recording only");
    }
    void wait() override {}
};
int main(int argc, char** argv) try {
    demand(argc == 2, "Expected authenticated 310.9.16 DLL path");
    const auto dll = std::filesystem::absolute(std::filesystem::u8path(argv[1]));
    phoenix_model::WeightSource source{dll, file_sha(dll)};
    Json results = Json::array();
    unsigned checks = 0;
    for (const char* model_id : {"original", "small-fp8"}) {
        source.model_id = model_id;
        Model model(source);
        const unsigned first = source.model_id == "small-fp8" ? 11 : 31;
        struct Profile {
            const char* name;
            bool attention, features, residual;
        };
        for (const auto& profile : std::array<Profile, 5>{{{"metal", false, true, true},
                                                           {"vulkan-scalar", false, true, true},
                                                           {"vulkan-fused", true, true, true},
                                                           {"hip", false, false, false},
                                                           {"sycl", false, false, false}}}) {
            std::array<unsigned, 8> baseline{};
            for (bool custom : {false, true}) {
                Recorder backend;
                const std::array<unsigned, 8> counts =
                    custom ? std::array<unsigned, 8>{0, 2, 1, 0, 4, 0, 0, 0}
                           : std::array<unsigned, 8>{1, 1, 1, 1, 1, 1, 1, 1};
                unsigned command_count = 0;
                {
                    Graph graph(backend, model, profile.attention, 128, profile.features,
                                profile.residual);
                    graph.build(16, 12, counts);
                    std::array<unsigned, 8> seen{};
                    for (const auto& command : graph.commands) {
                        ++checks;
                        demand(command.operation.family <= phx::FrameFeatures,
                               "Unsupported portable operator family");
                        if (!profile.attention)
                            demand(command.operation.family != phx::WindowAttention,
                                   "Unfused backend received fused attention");
                        if (command.block >= first && command.block < first + 8)
                            ++seen[command.block - first];
                    }
                    for (unsigned block = 0; block < 8; ++block) {
                        ++checks;
                        if (!custom) {
                            demand(seen[block] > 0, "Missing trained transformer");
                            baseline[block] = seen[block];
                        } else
                            demand(seen[block] == baseline[block] * counts[block],
                                   "Model-specific transformer skip/repetition mismatch");
                    }
                    command_count = unsigned(graph.commands.size());
                }
                ++checks;
                demand(backend.live == 0, "Recorded backend allocations leaked");
                results.push_back({{"model", model_id},
                                   {"backend_profile", profile.name},
                                   {"custom_repeats", custom},
                                   {"commands", command_count}});
            }
        }
    }
    std::cout << Json({{"passed", true},
                       {"checks", checks},
                       {"cases", results},
                       {"gpu_executed", false},
                       {"inference_executed", false}})
                     .dump()
              << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
