// Synthetic operator measurements, independent of a model or game capture.
#include "../../native_runtime/backend.h"
#include "../../native_runtime/backend_options.h"
#include "../../native_runtime/host_math.h"
#include "../../src/native/model/weight_source.h"
#include <chrono>
#include <iostream>
#include <algorithm>

using namespace phoenix_cpp;
using Json = nlohmann::json;
int wmain(int argc, wchar_t** argv) {
    try {
        demand(argc == 5 || argc == 6,
               "Usage: matrix_benchmark RUNTIME BACKEND DEVICE NEW_REPORT [OPTIONS]");
        demand(!std::filesystem::exists(argv[4]), "Fresh report required");
        const auto root = std::filesystem::absolute(argv[1]);
        const auto kind = std::filesystem::path(argv[2]).string();
        const auto device = std::filesystem::path(argv[3]).string();
        const uint32_t options = argc == 6 ? uint32_t(std::stoul(argv[5])) : 0;
        struct Libraries {
            HMODULE backend = nullptr, opencl = nullptr;
            DLL_DIRECTORY_COOKIE directory = nullptr;
            ~Libraries() {
                if (backend)
                    FreeLibrary(backend);
                if (opencl)
                    FreeLibrary(opencl);
                if (directory)
                    RemoveDllDirectory(directory);
            }
        } libraries;
        if (device == "sycl:cpu") {
            libraries.directory = AddDllDirectory((root / "opencl_cpu").c_str());
            libraries.opencl =
                LoadLibraryExW((root / "opencl_cpu/OpenCL.dll").c_str(), nullptr,
                               LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            demand(libraries.directory && libraries.opencl, "Private CPU ICD unavailable");
            SetEnvironmentVariableW(L"CL_CONFIG_CPU_TBB_NUM_WORKERS", L"32");
        }
        libraries.backend =
            LoadLibraryExW((root / ("phoenix_" + kind + "_cpp.dll")).c_str(), nullptr,
                           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        demand(libraries.backend, "Backend unavailable");
        using Configured = Backend* (*)(const char*, const wchar_t*, uint32_t, char*, size_t);
        auto configured = reinterpret_cast<Configured>(
            GetProcAddress(libraries.backend, "phoenix_backend_create_configured"));
        auto basic = reinterpret_cast<CreateBackend>(
            GetProcAddress(libraries.backend, "phoenix_backend_create"));
        auto destroy = reinterpret_cast<DestroyBackend>(
            GetProcAddress(libraries.backend, "phoenix_backend_destroy"));
        demand((configured || basic) && destroy, "Backend factory unavailable");
        char error[2048]{};
        std::unique_ptr<Backend, DestroyBackend> backend(
            configured ? configured(device.c_str(), (root / "vulkan").c_str(), options, error,
                                    sizeof error)
                       : basic(device.c_str(), (root / "vulkan").c_str(), error, sizeof error),
            destroy);
        demand(bool(backend), error);
        phx::Kernel<HostMath> converter{};
        Json report = {
            {"device", device},
            {"backend", kind},
            {"options", options},
            {"passed", false},
            {"rows", Json::array()},
            {"scope",
             "Synthetic resident matrix operators; two warmups and three timed samples; not frame time"}};
        struct Shape {
            unsigned rows, columns, reduction;
        };
        for (const auto shape :
             {Shape{8192, 128, 32}, Shape{8192, 32, 128}, Shape{32768, 512, 128},
              Shape{130560, 512, 128}, Shape{32640, 1024, 256}, Shape{32640, 256, 1024},
              Shape{2048, 1024, 256}, Shape{640, 1536, 512}, Shape{640, 4096, 1024},
              Shape{640, 1024, 4096}}) {
            std::vector<uint16_t> left(size_t(shape.rows) * shape.reduction);
            std::vector<uint8_t> right(size_t(shape.columns) * shape.reduction);
            std::vector<uint16_t> output(size_t(shape.rows) * shape.columns);
            for (size_t index = 0; index < left.size(); ++index)
                left[index] =
                    uint16_t(converter.half_bits(float(int(index * 13 % 23) - 11) * .0625f));
            for (size_t index = 0; index < right.size(); ++index)
                right[index] = uint8_t((32 + index * 17 % 57) | ((index & 1) ? 128 : 0));
            const auto left_buffer = backend->allocate(left.size() * 2),
                       right_buffer = backend->allocate(right.size()),
                       output_buffer = backend->allocate(output.size() * 2);
            backend->upload(left_buffer, left.data(), left.size() * 2);
            backend->upload(right_buffer, right.data(), right.size());
            phx::Operation operation{};
            operation.family = phx::MatrixMultiply;
            operation.flags = 1 | phx::RightInputFp8 | phx::LeftInputFp8Grid | phx::StoreHalf |
                              (phx::PublishHalfThenFp8 << 8);
            operation.left_input = left_buffer.address;
            operation.right_input = right_buffer.address;
            operation.output = output_buffer.address;
            operation.row_count = shape.rows;
            operation.column_count = shape.columns;
            operation.reduction_count = shape.reduction;
            operation.batch_count = 1;
            operation.left_row_stride = operation.right_row_stride = shape.reduction;
            operation.output_row_stride = shape.columns;
            std::vector<double> samples;
            for (unsigned iteration = 0; iteration < 5; ++iteration) {
                std::vector<Event> events;
                const auto started = std::chrono::steady_clock::now();
                backend->execute({{operation, output.size(), 0, 0}}, events);
                backend->wait();
                const double milliseconds = std::chrono::duration<double, std::milli>(
                                                std::chrono::steady_clock::now() - started)
                                                .count();
                if (iteration >= 2)
                    samples.push_back(milliseconds);
            }
            backend->download(output_buffer, output.data(), output.size() * 2);
            const auto checksum = phoenix_model::weight_digest(
                reinterpret_cast<const uint8_t*>(output.data()), output.size() * 2);
            auto sorted = samples;
            std::sort(sorted.begin(), sorted.end());
            report["rows"].push_back(
                {{"m", shape.rows},
                 {"n", shape.columns},
                 {"k", shape.reduction},
                 {"samples_ms", samples},
                 {"median_ms", sorted[1]},
                 {"owned_bytes", left_buffer.bytes + right_buffer.bytes + output_buffer.bytes},
                 {"output_sha256", checksum}});
            std::ofstream(argv[4]) << report.dump(2) << '\n';
            std::cout << shape.rows << 'x' << shape.columns << 'x' << shape.reduction << ": "
                      << sorted[1] << " ms\n"
                      << std::flush;
            backend->release(output_buffer);
            backend->release(right_buffer);
            backend->release(left_buffer);
        }
        report["passed"] = true;
        std::ofstream(argv[4]) << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
