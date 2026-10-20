// CPU recording only: scheduling must reuse immutable weights without re-decoding.
#include "../../native_runtime/graph.h"
#include "../../src/native/model/native_layout.h"
#include <iostream>
using namespace phoenix_cpp;
class Recorder final : public Backend {
    uint64_t address_ = 0x100000;
  public:
    uint64_t live = 0, uploads = 0;
    const char* name() const override { return "recording"; }
    Allocation allocate(uint64_t size) override {
        const auto address = address_; address_ += (size + 255) & ~uint64_t(255);
        live += size; return {phx::address_from_bits(address), size, address};
    }
    void release(Allocation value) override { demand(live >= value.bytes, "Double release"); live -= value.bytes; }
    void upload(Allocation value, const void*, uint64_t size) override {
        demand(size <= value.bytes, "Upload bounds"); ++uploads;
    }
    void download(Allocation, void*, uint64_t) override { throw std::runtime_error("No inference"); }
    void execute(const std::vector<Command>&, std::vector<Event>&) override { throw std::runtime_error("No inference"); }
    void wait() override {}
};
int main(int argc, char** argv) try {
    demand(argc == 2, "Expected authenticated model DLL");
    const auto dll = std::filesystem::absolute(std::filesystem::u8path(argv[1]));
    phoenix_model::WeightSource source{dll, file_sha(dll)};
    unsigned schedules = 0;
    for (const auto* id : {"original", "small-fp8"}) {
        source.model_id = id;
        Model model(source);
        Recorder backend;
        auto weights = std::make_shared<WeightStore>(backend, model);
        uint64_t uploaded = 0, resident = 0;
        for (unsigned step = 0; step < 5; ++step) {
            const unsigned depth = step % 2 ? 4 : 8;
            std::array<unsigned, 8> repeats{};
            for (unsigned i = 0; i < depth; ++i) repeats[i] = step == 3 ? 2 : 1;
            {
                Graph graph(backend, model, false, 128, false, false, false, weights);
                graph.build(16, 12, repeats);
                if (!step) { uploaded = backend.uploads; resident = graph.weight_bytes(); }
                demand(backend.uploads == uploaded, "Schedule re-uploaded immutable weights");
                demand(graph.weight_bytes() == resident, "Schedule duplicated resident weights");
                if (step) demand(!model.decoded(), "Schedule unnecessarily decoded model again");
                model.discard_decoded();
                ++schedules;
            }
            demand(backend.live == resident, "Schedule scratch allocations leaked");
        }
        Recorder different_backend;
        bool rejected = false;
        try { Graph wrong(different_backend, model, false, 128, false, false, false, weights); }
        catch (const std::exception&) { rejected = true; }
        demand(rejected, "Cross-backend weight cache accepted");
        weights.reset();
        demand(backend.live == 0, "Resident weights leaked after session destruction");
        // Begin with disabled transformers: enabling unseen weights may decode
        // once, but subsequent schedules still reuse the uploaded cache.
        weights = std::make_shared<WeightStore>(backend, model);
        {
            Graph graph(backend, model, false, 128, false, false, false, weights);
            graph.build(16, 12, {1,1,1,1,0,0,0,0});
        }
        model.discard_decoded();
        {
            Graph graph(backend, model, false, 128, false, false, false, weights);
            graph.build(16, 12, {1,1,1,1,1,1,1,1});
            demand(model.decoded(), "Unseen transformer weights were not decoded");
        }
        weights.reset();
        demand(backend.live == 0, "Lazily added weights leaked");
    }
    const phoenix_model::NativeSchedule future{88, 28, 3, 15, {1,0,0,0}, {}};
    demand(phoenix_model::NativeLayout{&future}.launches() == 88 &&
           phoenix_model::NativeLayout{&future}.transformer_begin() == 29,
           "Native layout is restricted to two architectures");
    std::cout << Json({{"passed", true}, {"gpu_executed", false}, {"cached_schedules", schedules}}).dump() << '\n';
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
