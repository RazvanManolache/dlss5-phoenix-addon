#include "../../native_runtime/broker_client.h"
#include <fstream>
#include <chrono>
#include <psapi.h>
#include <iostream>
using Json = nlohmann::json;
std::vector<float> plane(const std::filesystem::path& path, size_t elements) {
    std::vector<float> data(elements);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(data.data()), std::streamsize(elements * 4));
    if (!file || file.peek() != EOF)
        throw std::runtime_error("Plane extent mismatch");
    return data;
}
double milliseconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}
std::string output_hash(const std::vector<float>& pixels) {
    unsigned char digest[32];
    char hex[65]{};
    if (BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0,
                   reinterpret_cast<PUCHAR>(const_cast<float*>(pixels.data())),
                   ULONG(pixels.size() * sizeof(float)), digest, sizeof(digest)) < 0)
        throw std::runtime_error("Output hash failed");
    for (unsigned iteration_index = 0; iteration_index < 32; ++iteration_index)
        sprintf_s(hex + iteration_index * 2, 3, "%02x", unsigned(digest[iteration_index]));
    return hex;
}
Json memory(DWORD pid) {
    phoenix_cpp::Handle process(
        OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid));
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (!process.value ||
        !GetProcessMemoryInfo(process, reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                              sizeof(counters)))
        throw std::runtime_error("Process memory query failed");
    return {{"private_bytes", counters.PrivateUsage},
            {"working_set_bytes", counters.WorkingSetSize},
            {"peak_working_set_bytes", counters.PeakWorkingSetSize}};
}
int wmain(int argc, wchar_t** argv) {
    Json report = {{"passed", false}, {"cases", Json::array()}};
    std::filesystem::path out;
    try {
        if (argc != 3)
            throw std::runtime_error("Usage: runner.exe JOB.json NEW_OUTPUT");
        out = argv[2];
        if (std::filesystem::exists(out))
            throw std::runtime_error("Fresh output required");
        std::filesystem::create_directories(out);
        std::ifstream input(argv[1]);
        Json job;
        input >> job;
        auto config = job.at("config");
        unsigned width = job.at("width"), height = job.at("height");
        size_t pixels = size_t(width) * height;
        phoenix_cpp::NativeSession session;
        auto start = std::chrono::steady_clock::now();
        session.open(std::filesystem::u8path(config.at("runtime_root").get<std::string>()), config,
                     width, height);
        report["initialization_ms"] = milliseconds(start);
        report["device"] = config.at("device");
        report["width"] = width;
        report["height"] = height;
        report["runtime_pid"] = session.process_id();
        report["caller_pid"] = GetCurrentProcessId();
        report["initial_memory"] = memory(session.process_id());
        unsigned index = 0;
        for (const auto& task : job.at("cases")) {
            auto color =
                plane(std::filesystem::u8path(task.at("color").get<std::string>()), pixels * 3);
            std::vector<float> motion, depth, validity, output(pixels * 3);
            if (task.contains("motion"))
                motion = plane(std::filesystem::u8path(task.at("motion").get<std::string>()),
                               pixels * 2);
            if (task.contains("depth"))
                depth = plane(std::filesystem::u8path(task.at("depth").get<std::string>()), pixels);
            if (task.contains("validity"))
                validity =
                    plane(std::filesystem::u8path(task.at("validity").get<std::string>()), pixels);
            auto metadata = task.value("metadata", Json::object());
            metadata["has_motion"] = !motion.empty();
            metadata["has_depth"] = !depth.empty();
            metadata["has_history_validity"] = !validity.empty();
            start = std::chrono::steady_clock::now();
            int result = session.frame(color.data(), motion.empty() ? nullptr : motion.data(),
                                       depth.empty() ? nullptr : depth.data(),
                                       validity.empty() ? nullptr : validity.data(),
                                       metadata.dump(), output.data());
            auto wall = milliseconds(start);
            if (result)
                throw std::runtime_error(session.error());
            auto row = Json::parse(session.status_json());
            row["call_ms"] = wall;
            row["name"] = task.value("name", std::to_string(index));
            if (auto provider = GetModuleHandleW(L"phoenix_vulkan.dll")) {
                auto stats = reinterpret_cast<unsigned long long (*)(unsigned)>(
                    GetProcAddress(provider, "xmx_phoenix_memory"));
                auto mode =
                    reinterpret_cast<int (*)()>(GetProcAddress(provider, "xmx_staging_mode"));
                if (stats)
                    row["vulkan_transfer_buffer_bytes"] = stats(5);
                if (mode)
                    row["vulkan_staged_transfers"] = mode() != 0;
            }
            row["memory"] = memory(session.process_id());
            row["caller_memory"] = memory(GetCurrentProcessId());
            row["output_sha256"] = output_hash(output);
            if (task.contains("expected_sha256")) {
                row["baseline_exact"] = row["output_sha256"] == task["expected_sha256"];
                if (!row["baseline_exact"].get<bool>()) {
                    report["cases"].push_back(row);
                    std::ofstream failed(out / "failed_output.bin", std::ios::binary);
                    failed.write(reinterpret_cast<const char*>(output.data()),
                                 std::streamsize(output.size() * 4));
                    throw std::runtime_error("Output differs from saved per-backend baseline");
                }
            }
            if (task.value("save_output", true)) {
                std::ofstream file(out / (std::to_string(index) + ".bin"), std::ios::binary);
                file.write(reinterpret_cast<const char*>(output.data()),
                           std::streamsize(output.size() * 4));
                if (!file)
                    throw std::runtime_error("Output write failed");
            }
            report["cases"].push_back(row);
            ++index;
            std::ofstream snapshot(out / "report.json");
            snapshot << report.dump(2);
            std::cout << row["name"] << ": " << wall << " ms\n" << std::flush;
        }
        HMODULE modules[1024];
        DWORD bytes = 0;
        bool python = false;
        Json names = Json::array();
        phoenix_cpp::Handle audit_process(
            OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, session.process_id()));
        if (!audit_process.value ||
            !EnumProcessModules(audit_process, modules, sizeof(modules), &bytes) ||
            bytes > sizeof(modules))
            throw std::runtime_error("Module audit failed");
        for (unsigned iteration_index = 0; iteration_index < bytes / sizeof(HMODULE);
             ++iteration_index) {
            char name[32768];
            GetModuleFileNameExA(audit_process, modules[iteration_index], name, sizeof(name));
            auto filename = std::filesystem::path(name).filename().string();
            std::transform(filename.begin(), filename.end(), filename.begin(),
                           [](unsigned char character) { return char(std::tolower(character)); });
            if (filename.starts_with("python") || filename == "torch.dll" ||
                filename == "torch_cpu.dll" || filename == "c10.dll")
                python = true;
            names.push_back(filename);
        }
        report["python_or_torch_loaded"] = python;
        report["modules"] = names;
        report["passed"] = !python;
    } catch (const std::exception& exception) {
        report["error"] = exception.what();
        std::cerr << exception.what() << '\n';
    }
    if (!out.empty() && std::filesystem::is_directory(out)) {
        std::ofstream file(out / "report.json");
        file << report.dump(2);
    }
    return report["passed"].get<bool>() ? 0 : 1;
}
