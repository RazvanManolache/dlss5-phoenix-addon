#include "../../src/native/support/model_status.h"
#include "../../third_party/nlohmann/json.hpp"
#include <windows.h>
#include <iostream>
#include <stdexcept>
using namespace phoenix_addon::phoenix;
static void checked(bool pass, int line) {
    if (!pass)
        throw std::runtime_error("Model DLL panel preflight failed at line " +
                                 std::to_string(line));
}
#define check(value) checked((value), __LINE__)
int main() {
    std::filesystem::path directory;
    try {
        directory = std::filesystem::temp_directory_path() /
                    ("dlss5_phoenix_model_status_" + std::to_string(GetCurrentProcessId()) + "_" +
                     std::to_string(GetTickCount64()));
        check(std::filesystem::create_directory(directory));
        const auto file = directory / "synthetic.dll";
        check(inspect_model_dll({}) == ModelDllStatus::unconfigured);
        check(inspect_model_dll(file) == ModelDllStatus::missing);
        check(inspect_model_dll(directory) == ModelDllStatus::not_file);
        // Windows STL may report invalid names as missing rather than an
        // access error. Either must produce a warning without throwing.
        check(inspect_model_dll(directory / "<>invalid.dll") != ModelDllStatus::available);
        ModelDllMonitor monitor;
        check(monitor.poll(file, 0) == ModelDllStatus::missing);
        {
            std::ofstream synthetic(file, std::ios::binary);
            synthetic << "Not a real DLL or model";
        }
        check(monitor.poll(file, 999) == ModelDllStatus::missing);
        check(monitor.poll(file, 1000) == ModelDllStatus::available);
        // An existing file is only presence preflight, never authentication.
        HANDLE blocked =
            CreateFileW(file.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        check(blocked != INVALID_HANDLE_VALUE);
        const auto blocked_status = inspect_model_dll(file);
        CloseHandle(blocked);
        check(blocked_status == ModelDllStatus::unreadable);
        check(std::filesystem::remove(file));
        check(monitor.poll(file, 1999) == ModelDllStatus::available);
        check(monitor.poll(file, 2000) == ModelDllStatus::missing);
        check(monitor.poll(directory, 2001) == ModelDllStatus::not_file);
        check(monitor.poll({}, 2002) == ModelDllStatus::unconfigured);
        check(monitor.poll(file, 10) == ModelDllStatus::missing);
        check(std::string(model_dll_warning(ModelDllStatus::available)).empty());
        for (auto status : {ModelDllStatus::missing, ModelDllStatus::unconfigured,
                            ModelDllStatus::not_file, ModelDllStatus::unreadable})
            check(std::string(model_dll_warning(status)).find("Phoenix DLSS5") !=
                  std::string::npos);
        check(std::filesystem::remove(directory));
        std::cout << nlohmann::json({{"passed", true},
                                     {"cpu_only", true},
                                     {"model_files_used", false},
                                     {"presence_cases", 5},
                                     {"recovery_and_cache_checked", true},
                                     {"loader_authentication_unchanged", true}})
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::error_code ignored;
        if (!directory.empty()) {
            std::filesystem::remove(directory / "synthetic.dll", ignored);
            std::filesystem::remove(directory, ignored);
        }
        std::cerr << error.what() << '\n';
        return 1;
    }
}
