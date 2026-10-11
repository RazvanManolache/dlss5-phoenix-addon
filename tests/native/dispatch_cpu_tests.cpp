#include "../../src/native/support/dispatch.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <new>

namespace allocations {
thread_local bool enabled = false;
thread_local uint64_t created = 0, freed = 0, bytes = 0;
void add(size_t size) noexcept {
    if (enabled) {
        ++created;
        bytes += size;
    }
}
void remove(void* ptr) noexcept {
    if (enabled && ptr)
        ++freed;
}
}
void* operator new(size_t size) {
    void* ptr = std::malloc(size ? size : 1);
    if (!ptr)
        throw std::bad_alloc();
    allocations::add(size);
    return ptr;
}
void operator delete(void* ptr) noexcept {
    allocations::remove(ptr);
    std::free(ptr);
}
void operator delete(void* ptr, size_t) noexcept {
    ::operator delete(ptr);
}
void* operator new[](size_t size) {
    return ::operator new(size);
}
void operator delete[](void* ptr) noexcept {
    ::operator delete(ptr);
}
void operator delete[](void* ptr, size_t) noexcept {
    ::operator delete(ptr);
}
void* operator new(size_t size, std::align_val_t alignment) {
    void* ptr = _aligned_malloc(size ? size : 1, static_cast<size_t>(alignment));
    if (!ptr)
        throw std::bad_alloc();
    allocations::add(size);
    return ptr;
}
void operator delete(void* ptr, std::align_val_t) noexcept {
    allocations::remove(ptr);
    _aligned_free(ptr);
}
void operator delete(void* ptr, size_t, std::align_val_t alignment) noexcept {
    ::operator delete(ptr, alignment);
}
void* operator new[](size_t size, std::align_val_t alignment) {
    return ::operator new(size, alignment);
}
void operator delete[](void* ptr, std::align_val_t alignment) noexcept {
    ::operator delete(ptr, alignment);
}
void operator delete[](void* ptr, size_t, std::align_val_t alignment) noexcept {
    ::operator delete(ptr, alignment);
}

using namespace phoenix_addon;
using Launch = NVAPI_CU_KERNEL_LAUNCH_PARAMS;
using Dispatch = decltype(&NvAPI_D3D12_LaunchCuKernelChain);
using Json = phoenix_bundle::Json;

// Frozen pre-optimization body from a57g/a59g dispatch.h, SHA256
// 3705F61F658527D97A07C1547F55AF1102E4C2782E0F63D88E8B6D580EE0EF5B.
// Only API construction is replaced by injection; the dispatch body is verbatim.
class Reference {
    Dispatch real_;
    inline static thread_local Reference* active_ = nullptr;
    ID3D12GraphicsCommandList* command_ = nullptr;
    ResolvedControls controls_{};
    unsigned position_ = 0;
    uint64_t launches_ = 0;
    static NvAPI_Status __cdecl dispatch(ID3D12GraphicsCommandList* cmd,
                                         const NVAPI_CU_KERNEL_LAUNCH_PARAMS* source, NvU32 count) {
        auto* self = active_;
        check(self && cmd == self->command_ && source && count && count <= 154 - self->position_,
              "phoenix dispatch outside verified frame scope");
        std::vector<NVAPI_CU_KERNEL_LAUNCH_PARAMS> launches(source, source + count);
        std::array<uint8_t, 264> stem{};
        std::array<uint8_t, 184> head{};
        for (NvU32 iteration_index = 0; iteration_index < count; ++iteration_index) {
            const unsigned position = self->position_ + iteration_index;
            auto& launch = launches[iteration_index];
            if (position == 0) {
                check(launch.pParams && launch.paramSize == stem.size(),
                      "phoenix missing stem ABI");
                std::memcpy(stem.data(), launch.pParams, stem.size());
                patch_stem(stem.data(), stem.size(), self->controls_);
                launch.pParams = stem.data();
            } else if (position == 153) {
                check(launch.pParams && launch.paramSize == head.size(),
                      "phoenix missing head ABI");
                std::memcpy(head.data(), launch.pParams, head.size());
                patch_head(head.data(), head.size(), self->controls_);
                launch.pParams = head.data();
            }
        }
        dispatch_audit::OwnCall own_dispatch;
        const auto result = self->real_(cmd, launches.data(), count);
        self->position_ += count;
        self->launches_ += count;
        return result;
    }

  public:
    explicit Reference(Dispatch callback) : real_(callback) {}
    static Dispatch dispatch_function() noexcept {
        return dispatch;
    }
    uint64_t launches() const noexcept {
        return launches_;
    }
    class Scope {
        Reference& owner_;

      public:
        Scope(Reference& owner, ID3D12GraphicsCommandList* cmd, const ResolvedControls& controls)
            : owner_(owner) {
            check(!active_, "phoenix nested candidate dispatch");
            owner.command_ = cmd;
            owner.controls_ = controls;
            owner.position_ = 0;
            active_ = &owner;
        }
        ~Scope() {
            active_ = nullptr;
            owner_.command_ = nullptr;
        }
        void finish() const {
            check(owner_.position_ == 154, "phoenix incomplete candidate frame");
        }
    };
};

struct Fixture {
    std::array<Launch, 154> launches{};
    std::array<std::array<uint8_t, 264>, 154> arguments{};
    explicit Fixture(unsigned parity = 0) {
        for (unsigned iteration_index = 0; iteration_index < launches.size(); ++iteration_index) {
            auto& launch = launches[iteration_index];
            launch.hFunction = reinterpret_cast<NVDX_ObjectHandle>(
                uintptr_t(0x1000 + iteration_index * 16 + parity));
            launch.gridDim = {iteration_index + 1, iteration_index % 7 + 1, 1};
            launch.blockDim = {32, iteration_index % 4 + 1, 1};
            launch.dynSharedMemBytes = iteration_index * 128;
            launch.paramSize = iteration_index == 0
                                   ? 264
                                   : (iteration_index == 153 ? 184 : 32 + iteration_index % 97);
            for (unsigned inner_index = 0; inner_index < arguments[iteration_index].size();
                 ++inner_index)
                arguments[iteration_index][inner_index] =
                    uint8_t((iteration_index * 19 + inner_index * 7 + parity * 61) & 255);
            launch.pParams = arguments[iteration_index].data();
        }
    }
};
struct Trace {
    std::array<Launch, 154> launches{};
    std::array<std::array<uint8_t, 264>, 154> arguments{};
    std::array<unsigned, 154> call_sizes{};
    unsigned size = 0, calls = 0;
    NvAPI_Status status = NVAPI_OK;
    bool throw_in_dispatch = false;
};
thread_local Trace* trace = nullptr;
volatile uint64_t checksum = 0;
ID3D12GraphicsCommandList* command() {
    return reinterpret_cast<ID3D12GraphicsCommandList*>(uintptr_t(0x1234));
}
__declspec(noinline) NvAPI_Status __cdecl capture(ID3D12GraphicsCommandList* cmd,
                                                  const Launch* source, NvU32 count) {
    check(cmd == command() && trace && source, "fake dispatch arguments");
    check(dispatch_audit::own_call, "fake dispatch missing OwnCall");
    dispatch_audit::Call audit(count);
    if (trace->throw_in_dispatch)
        throw std::runtime_error("fake dispatch exception");
    check(trace->size + count <= 154 && trace->calls < 154, "fake trace overflow");
    trace->call_sizes[trace->calls++] = count;
    for (unsigned iteration_index = 0; iteration_index < count; ++iteration_index) {
        const auto index = trace->size++;
        check(source[iteration_index].paramSize <= 264 && source[iteration_index].pParams,
              "fake parameter extent");
        trace->launches[index] = source[iteration_index];
        std::memcpy(trace->arguments[index].data(), source[iteration_index].pParams,
                    source[iteration_index].paramSize);
    }
    if (trace->status == NVAPI_OK)
        audit.accepted();
    return trace->status;
}
__declspec(noinline) NvAPI_Status __cdecl consume(ID3D12GraphicsCommandList*, const Launch* source,
                                                  NvU32 count) {
    const auto* bytes = static_cast<const uint8_t*>(source[0].pParams);
    checksum = checksum + source[0].gridDim.x + source[0].paramSize + bytes[0] + count;
    return NVAPI_OK;
}
bool equal_launch(const Launch& actual, const Launch& expected, bool pointers = false) {
    return actual.hFunction == expected.hFunction && actual.gridDim.x == expected.gridDim.x &&
           actual.gridDim.y == expected.gridDim.y && actual.gridDim.z == expected.gridDim.z &&
           actual.blockDim.x == expected.blockDim.x && actual.blockDim.y == expected.blockDim.y &&
           actual.blockDim.z == expected.blockDim.z &&
           actual.dynSharedMemBytes == expected.dynSharedMemBytes &&
           actual.paramSize == expected.paramSize &&
           (!pointers || actual.pParams == expected.pParams);
}
void equal_trace(const Trace& actual, const Trace& expected) {
    check(actual.calls == expected.calls && actual.size == expected.size &&
              actual.call_sizes == expected.call_sizes,
          "dispatch shape differs");
    for (unsigned iteration_index = 0; iteration_index < actual.size; ++iteration_index) {
        check(equal_launch(actual.launches[iteration_index], expected.launches[iteration_index]),
              "descriptor field differs");
        check(std::memcmp(actual.arguments[iteration_index].data(),
                          expected.arguments[iteration_index].data(),
                          actual.launches[iteration_index].paramSize) == 0,
              "argument bytes differ");
    }
}
void equal_counts(const dispatch_audit::Counts& actual, const dispatch_audit::Counts& expected) {
    check(actual.own_calls == expected.own_calls && actual.other_calls == expected.other_calls &&
              actual.own_attempted == expected.own_attempted &&
              actual.own_accepted == expected.own_accepted &&
              actual.other_attempted == expected.other_attempted &&
              actual.other_accepted == expected.other_accepted,
          "audit counts differ");
}
struct Result {
    Trace trace;
    dispatch_audit::Counts counts;
    uint64_t launches = 0;
};
template <class Wrapper>
Result run(Fixture& input, const ResolvedControls& controls, const std::vector<unsigned>& schedule,
           NvAPI_Status status) {
    Result result;
    result.trace.status = status;
    trace = &result.trace;
    const auto original_launches = input.launches;
    const auto original_arguments = input.arguments;
    Wrapper wrapper(capture);
    {
        dispatch_audit::Scope audit(result.counts);
        typename Wrapper::Scope scope(wrapper, command(), controls);
        unsigned position = 0;
        for (unsigned count : schedule) {
            check(Wrapper::dispatch_function()(command(), input.launches.data() + position,
                                               count) == status,
                  "status propagation differs");
            position += count;
        }
        scope.finish();
    }
    result.launches = wrapper.launches();
    check(!dispatch_audit::own_call && !dispatch_audit::active, "scope leaked audit state");
    check(input.arguments == original_arguments, "source argument bytes mutated");
    for (unsigned iteration_index = 0; iteration_index < input.launches.size(); ++iteration_index)
        check(
            equal_launch(input.launches[iteration_index], original_launches[iteration_index], true),
            "source descriptor mutated");
    trace = nullptr;
    return result;
}

template <class Function> std::string failure(Function&& callback) {
    try {
        callback();
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    throw std::runtime_error("expected rejection did not happen");
}
enum class Fault {
    no_scope,
    wrong_command,
    null_source,
    zero_count,
    too_many,
    bad_stem,
    null_stem,
    bad_head,
    null_head,
    incomplete,
    nested,
    after_complete,
    dispatch_throw
};
struct Failed {
    Result result;
    std::string message;
    bool finish_failed = false;
};
template <class Wrapper> Failed invalid(Fault fault) {
    Fixture input;
    Failed result;
    trace = &result.result.trace;
    Wrapper wrapper(capture);
    auto call = [&](ID3D12GraphicsCommandList* cmd, const Launch* launch, unsigned count) {
        return Wrapper::dispatch_function()(cmd, launch, count);
    };
    dispatch_audit::Scope audit(result.result.counts);
    if (fault == Fault::no_scope)
        result.message = failure([&] { call(command(), input.launches.data(), 1); });
    else {
        typename Wrapper::Scope scope(wrapper, command(), ResolvedControls{});
        result.message = failure([&] {
            switch (fault) {
            case Fault::wrong_command:
                call(nullptr, input.launches.data(), 1);
                break;
            case Fault::null_source:
                call(command(), nullptr, 1);
                break;
            case Fault::zero_count:
                call(command(), input.launches.data(), 0);
                break;
            case Fault::too_many:
                call(command(), input.launches.data(), 155);
                break;
            case Fault::bad_stem:
                --input.launches[0].paramSize;
                call(command(), input.launches.data(), 1);
                break;
            case Fault::null_stem:
                input.launches[0].pParams = nullptr;
                call(command(), input.launches.data(), 1);
                break;
            case Fault::bad_head:
            case Fault::null_head:
                call(command(), input.launches.data(), 153);
                if (fault == Fault::bad_head)
                    --input.launches[153].paramSize;
                else
                    input.launches[153].pParams = nullptr;
                call(command(), &input.launches[153], 1);
                break;
            case Fault::incomplete:
                call(command(), input.launches.data(), 1);
                scope.finish();
                break;
            case Fault::nested: {
                typename Wrapper::Scope nested(wrapper, command(), ResolvedControls{});
                break;
            }
            case Fault::after_complete:
                call(command(), input.launches.data(), 154);
                call(command(), input.launches.data(), 1);
                break;
            case Fault::dispatch_throw:
                result.result.trace.throw_in_dispatch = true;
                call(command(), input.launches.data(), 1);
                break;
            default:
                throw std::runtime_error("unexpected fault");
            }
        });
        try {
            scope.finish();
        } catch (const std::runtime_error&) {
            result.finish_failed = true;
        }
    }
    result.result.launches = wrapper.launches();
    check(!dispatch_audit::own_call, "OwnCall leaked after failure");
    trace = nullptr;
    return result;
}

template <class Wrapper>
void frames(Wrapper& wrapper, Fixture& input, const ResolvedControls& controls, unsigned count) {
    for (unsigned frame = 0; frame < count; ++frame) {
        typename Wrapper::Scope scope(wrapper, command(), controls);
        for (const auto& launch : input.launches)
            check(Wrapper::dispatch_function()(command(), &launch, 1) == NVAPI_OK,
                  "benchmark status");
        scope.finish();
    }
}
template <class Wrapper> Json allocation_test(Fixture& input, const ResolvedControls& controls) {
    Wrapper wrapper(consume);
    frames(wrapper, input, controls, 10);
    allocations::created = allocations::freed = allocations::bytes = 0;
    allocations::enabled = true;
    try {
        frames(wrapper, input, controls, 100);
    } catch (...) {
        allocations::enabled = false;
        throw;
    }
    allocations::enabled = false;
    return {{"frames", 100},
            {"allocations", allocations::created},
            {"frees", allocations::freed},
            {"allocated_bytes", allocations::bytes}};
}
template <class Wrapper>
double timed(Wrapper& wrapper, Fixture& input, const ResolvedControls& controls, unsigned count) {
    const auto start = std::chrono::steady_clock::now();
    frames(wrapper, input, controls, count);
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / count;
}
Json distribution(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return {{"median", (samples[9] + samples[10]) / 2},
            {"min", samples.front()},
            {"max", samples.back()},
            {"p25", samples[5]},
            {"p75", samples[14]}};
}
int main() {
    try {
        check(!GetModuleHandleW(L"nvapi64.dll") && !GetModuleHandleW(L"nvngx_dlssnr.dll"),
              "GPU API already loaded");
        const std::vector<std::vector<unsigned>> schedules{
            std::vector<unsigned>(154, 1),
            {1, 4, 4, 6, 8, 33, 1, 40, 1, 33, 8, 6, 4, 4, 1},
            {154},
            {1, 152, 1}};
        unsigned cases = 0;
        for (unsigned width : {640u, 2560u})
            for (unsigned parity : {0u, 1u})
                for (uint64_t sampling : {uint64_t(0), uint64_t(1), uint64_t(0x100000007)})
                    for (bool history : {false, true})
                        for (bool reset : {false, true})
                            for (bool motion : {false, true}) {
                                Fixture input(parity);
                                Controls host_controls;
                                host_controls.reset = reset;
                                host_controls.motion_x = -2.25f;
                                host_controls.motion_y = 3.5f;
                                host_controls.tone = 0.625f;
                                host_controls.structure = 1.75f;
                                host_controls.auto_mask = 1;
                                host_controls.skin = parity ? -1.0f : 0.375f;
                                const auto resolved_controls = resolve_controls(
                                    host_controls, width, width == 640 ? 384 : 1080, motion,
                                    history, sampling);
                                for (const auto& schedule : schedules)
                                    for (auto status : {NVAPI_OK, NVAPI_ERROR}) {
                                        const auto expected = run<Reference>(
                                            input, resolved_controls, schedule, status);
                                        const auto actual = run<FrameDispatch>(
                                            input, resolved_controls, schedule, status);
                                        equal_trace(expected.trace, actual.trace);
                                        equal_counts(expected.counts, actual.counts);
                                        check(expected.launches == 154 && actual.launches == 154,
                                              "launch total differs");
                                        check(actual.counts.own_calls == schedule.size() &&
                                                  actual.counts.own_attempted == 154 &&
                                                  actual.counts.own_accepted ==
                                                      (status == NVAPI_OK ? 154u : 0u) &&
                                                  !actual.counts.other_calls,
                                              "wrong dispatch audit totals");
                                        ++cases;
                                    }
                            }
        for (auto fault :
             {Fault::no_scope, Fault::wrong_command, Fault::null_source, Fault::zero_count,
              Fault::too_many, Fault::bad_stem, Fault::null_stem, Fault::bad_head, Fault::null_head,
              Fault::incomplete, Fault::nested, Fault::after_complete, Fault::dispatch_throw}) {
            const auto expected = invalid<Reference>(fault);
            const auto actual = invalid<FrameDispatch>(fault);
            check(expected.message == actual.message &&
                      expected.finish_failed == actual.finish_failed &&
                      expected.result.launches == actual.result.launches,
                  "failure/counter behavior differs");
            equal_trace(expected.result.trace, actual.result.trace);
            equal_counts(expected.result.counts, actual.result.counts);
            ++cases;
        }
        check(failure([] { FrameDispatch invalid(nullptr); }) == "phoenix null injected dispatch",
              "null injection accepted");
        ++cases;
        Fixture fixture;
        const auto repeat_controls = resolve_controls(Controls{}, 640, 384, true, true, 42);
        unsigned depth_cases = 0;
        for (unsigned depth = 1; depth <= 8; ++depth)
            for (const auto& schedule : schedules) {
                Fixture input;
                for (unsigned block_index = 0; block_index < 8; ++block_index)
                    input.launches[61 + 5 * block_index].paramSize = 72;
                input.launches[97].paramSize = 24;
                const auto before = input.arguments;
                FrameDispatch wrapper(capture);
                Trace actual;
                trace = &actual;
                {
                    FrameDispatch::Scope scope(wrapper, command(), repeat_controls, depth);
                    unsigned position = 0;
                    for (auto count : schedule) {
                        check(FrameDispatch::dispatch_function()(
                                  command(), input.launches.data() + position, count) == NVAPI_OK,
                              "depth dispatch failed");
                        position += count;
                    }
                    scope.finish();
                }
                trace = nullptr;
                const unsigned omitted = 5 * (8 - depth);
                check(actual.size == 154 - omitted && wrapper.expected_launches() == 154 - omitted,
                      "depth launch count");
                for (unsigned original = 1, at = 1; original < 153; ++original) {
                    if (original >= 57 + 5 * depth && original < 97)
                        continue;
                    check(equal_launch(input.launches[original], actual.launches[at]),
                          "depth changed retained launch descriptor");
                    auto expected = input.arguments[original];
                    if (depth < 8 && original == 97)
                        std::memcpy(expected.data(), input.arguments[56 + 5 * depth].data() + 16,
                                    8);
                    check(std::memcmp(expected.data(), actual.arguments[at].data(),
                                      actual.launches[at].paramSize) == 0,
                          "depth pointer or retained bytes differ");
                    ++at;
                }
                check(input.arguments == before, "depth mutated source arguments");
                ++depth_cases;
            }
        for (auto depth : {0u, 9u}) {
            FrameDispatch wrapper(capture);
            check(!failure([&] {
                       FrameDispatch::Scope scope(wrapper, command(), repeat_controls, depth);
                   }).empty(),
                  "bad depth accepted");
            ++depth_cases;
        }
        {
            FrameDispatch wrapper(capture);
            skip_groups_mask.store(1);
            check(!failure([&] {
                       FrameDispatch::Scope scope(wrapper, command(), repeat_controls, 4);
                   }).empty(),
                  "stale ablation combined with depth");
            skip_groups_mask.store(0);
            ++depth_cases;
        }
        auto old_allocations = allocation_test<Reference>(fixture, repeat_controls);
        auto new_allocations = allocation_test<FrameDispatch>(fixture, repeat_controls);
        check(old_allocations["allocations"] == 15400 && old_allocations["frees"] == 15400,
              "reference allocation observation changed");
        check(new_allocations["allocations"] == 0 && new_allocations["frees"] == 0,
              "fast path allocates");
        Reference old_wrapper(consume);
        FrameDispatch new_wrapper(consume);
        frames(old_wrapper, fixture, repeat_controls, 1000);
        frames(new_wrapper, fixture, repeat_controls, 1000);
        Json batches = Json::array();
        std::vector<double> old_times, new_times;
        for (unsigned batch = 0; batch < 20; ++batch) {
            double old_time, new_time;
            if (batch % 2 == 0) {
                old_time = timed(old_wrapper, fixture, repeat_controls, 10000);
                new_time = timed(new_wrapper, fixture, repeat_controls, 10000);
            } else {
                new_time = timed(new_wrapper, fixture, repeat_controls, 10000);
                old_time = timed(old_wrapper, fixture, repeat_controls, 10000);
            }
            old_times.push_back(old_time);
            new_times.push_back(new_time);
            batches.push_back(
                {{"index", batch},
                 {"order", batch % 2 == 0 ? "reference_then_actual" : "actual_then_reference"},
                 {"reference_us_per_frame", old_time},
                 {"actual_us_per_frame", new_time}});
        }
        check(!GetModuleHandleW(L"nvapi64.dll") && !GetModuleHandleW(L"nvngx_dlssnr.dll"),
              "GPU API loaded during CPU tests");
        Json report{{"passed", true},
                    {"cpu_only", true},
                    {"real_header_under_test", "production_game_phoenix_dispatch.h"},
                    {"differential_cases", cases},
                    {"nvapi_loaded", false},
                    {"dlssnr_loaded", false},
                    {"depth_cases", depth_cases},
                    {"reference_allocations", old_allocations},
                    {"actual_allocations", new_allocations},
                    {"timing_scope",
                     "CPU wrapper, fake non-inlined dispatch; not NVAPI or GPU/game performance"},
                    {"frames_per_batch", 10000},
                    {"launches_per_frame", 154},
                    {"batches", batches},
                    {"reference_us_per_frame", distribution(old_times)},
                    {"actual_us_per_frame", distribution(new_times)},
                    {"checksum", uint64_t(checksum)}};
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        allocations::enabled = false;
        std::cout << Json{{"passed", false}, {"cpu_only", true}, {"error", error.what()}}.dump(2)
                  << '\n';
        return 1;
    }
}
