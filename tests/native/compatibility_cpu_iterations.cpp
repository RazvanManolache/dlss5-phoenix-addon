// GPU integration test; compile-only source builds must not execute it.
// Run with an external model config and a matching embedded v10 helper/core.
#include "../../src/integration/compatibility/host_client.h"
#include <iostream>
using namespace phoenix_compat;

static void await_ready(HostClient& host) {
    const auto until = GetTickCount64() + 90000;
    while (host.phase() != Phase::Ready && GetTickCount64() < until) {
        require(host.phase() != Phase::Failed, host.error().c_str());
        Sleep(2);
    }
    require(host.phase() == Phase::Ready, "CPU iteration helper startup timed out");
}
static std::vector<uint8_t> frame(HostClient& host, uint64_t serial, bool reset,
                                  const std::vector<uint8_t>& input, uint32_t iterations) {
    await_ready(host);
    require(host.submit(serial, reset, input, iterations), "CPU iteration submit failed");
    const auto until = GetTickCount64() + 90000;
    while (host.phase() != Phase::Finished && GetTickCount64() < until) {
        require(host.phase() != Phase::Failed, host.error().c_str());
        Sleep(2);
    }
    require(host.phase() == Phase::Finished, "CPU iteration result timed out");
    return host.take();
}
static void await_exit(DWORD pid) {
    Handle process;
    process.value = OpenProcess(SYNCHRONIZE, FALSE, pid);
    if (process.value)
        require(WaitForSingleObject(process.value, 10000) == WAIT_OBJECT_0,
                "Previous helper did not terminate");
}

int main(int argc, char** argv) try {
    require(argc == 2, "usage: cpu_iterations weight_source.json");
    Json config;
    std::ifstream(argv[1]) >> config;
    auto source = phoenix_model::WeightSource::config(
        config.contains("weight_source") ? config.at("weight_source") : config);
    phoenix_addon::phoenix::Settings settings;
    settings.mode = phoenix_addon::phoenix::ours;
    settings.refinement_passes = 1; // Distinct from the helper iterations under test.
    constexpr unsigned width = 257, height = 193;
    FeedBuild build{};
    build.width = build.output_width = width;
    build.height = build.output_height = height;
    build.color_fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
    build.output_fmt = DXGI_FORMAT_R16G16B16A16_FLOAT;
    build.cpu_bridge = 1;
    build.mv_scale_x = build.mv_scale_y = 1;
    std::vector<uint8_t> input(size_t(width) * height * 4);
    for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x) {
            const auto pixel = (size_t(y) * width + x) * 4;
            input[pixel] = uint8_t(x * 255 / width);
            input[pixel + 1] = uint8_t(y * 255 / height);
            input[pixel + 2] = uint8_t(((x / 8) ^ (y / 8)) & 1 ? 180 : 70);
            input[pixel + 3] = 255;
        }
    std::vector<uint8_t> expected, actual;
    DWORD reference_pid = 0, candidate_pid = 0;
    {
        HostClient reference(GetModuleHandleW(nullptr), source, settings, build);
        auto quantized_first = frame(reference, 1, true, input, 1);
        require(quantized_first.size() == input.size(), "Reference returned wrong byte extent");
        expected = frame(reference, 2, false, quantized_first, 1);
        reference_pid = reference.process_id();
    }
    await_exit(reference_pid);
    {
        HostClient candidate(GetModuleHandleW(nullptr), source, settings, build);
        actual = frame(candidate, 1, true, input, 2);
        candidate_pid = candidate.process_id();
    }
    await_exit(candidate_pid);
    require(actual == expected, "FP16 two-pass rounding differs from explicit RGBA8 roundtrip");
    require(actual != input, "Test did not produce a model-modified image");

    // Scaling belongs inside the core; unequal transport canvases must fail at
    // Build rather than execute a second pass with the wrong texture geometry.
    build.output_width = width * 2;
    build.output_height = height * 2;
    DWORD rejected_pid = 0;
    {
        HostClient rejected(GetModuleHandleW(nullptr), source, settings, build);
        const auto until = GetTickCount64() + 90000;
        while (rejected.phase() != Phase::Failed && GetTickCount64() < until) {
            require(rejected.phase() != Phase::Ready,
                    "Unequal CPU transport canvases were accepted");
            Sleep(2);
        }
        require(rejected.phase() == Phase::Failed &&
                    rejected.error().find("rejected frame resources") != std::string::npos,
                "Unequal canvas test failed before reaching the Build rejection");
        rejected_pid = rejected.process_id();
    }
    await_exit(rejected_pid);
    std::cout << Json({{"passed", true},
                       {"gpu_executed", true},
                       {"client_bits", sizeof(void*) * 8},
                       {"fp16_cpu_two_pass_exact", true},
                       {"unequal_canvas_rejected", true},
                       {"width", width},
                       {"height", height},
                       {"output_sha256",
                        phoenix_model::weight_digest(actual.data(), actual.size())}})
                     .dump()
              << '\n';
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
