#include "runtime_api.h"
#include "../third_party/nlohmann/json.hpp"
#include <iostream>
#include <fstream>
using Json = nlohmann::json;
Json entries = Json::array();
void device(const char* device_id, const char* name) {
    std::string key = device_id;
    if (key.rfind("cuda:", 0) == 0 && key.size() == 37) {
        auto hex = key.substr(5);
        key = "cuda:GPU-" + hex.substr(0, 8) + "-" + hex.substr(8, 4) + "-" + hex.substr(12, 4) +
              "-" + hex.substr(16, 4) + "-" + hex.substr(20);
    }
    entries.push_back({{"id", key}, {"label", name}});
}
int wmain(int argc, wchar_t** argv) {
    try {
        if (argc < 2 || argc > 3)
            throw std::runtime_error("Usage: device_query.exe RUNTIME_ROOT [private-cpu]");
        if (argc == 3 && std::wstring(argv[2]) != L"private-cpu")
            throw std::runtime_error("Unknown option; expected private-cpu");
        // DLL-directory search flags require an absolute package path on Windows.
        auto root = std::filesystem::absolute(std::filesystem::path(argv[1]));
        if (!std::filesystem::is_directory(root))
            throw std::runtime_error("Runtime package directory does not exist");
        DLL_DIRECTORY_COOKIE cookie = nullptr;
        if (argc == 3) {
            auto folder = root / "opencl_cpu";
            auto icd = folder / "intelocl64.dll";
            if (!std::filesystem::is_regular_file(icd) ||
                !std::filesystem::is_regular_file(folder / "OpenCL.dll"))
                throw std::runtime_error(
                    "private-cpu requires a packaged Intel CPU ICD and OpenCL proxy in opencl_cpu; "
                    "omit this option for CPU/CUDA/Vulkan packages");
            cookie = AddDllDirectory(folder.c_str());
            if (!cookie || !SetEnvironmentVariableW(L"OCL_ICD_FILENAMES", icd.c_str()))
                throw std::runtime_error("CPU ICD query setup failed");
            SetEnvironmentVariableW(L"CL_CONFIG_CPU_TBB_NUM_WORKERS", L"8");
        }
        Json report = {{"devices", Json::array({{{"id", "cpu"}, {"label", "CPU C++"}}})},
                       {"errors", Json::object()}};
        for (const auto& kind : {"cuda", "hip", "sycl"}) {
            auto path = root / (std::string("phoenix_") + kind + "_cpp.dll");
            auto library =
                LoadLibraryExW(path.c_str(), nullptr,
                               LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if (!library) {
                report["errors"][kind] =
                    "Library unavailable (Windows " + std::to_string(GetLastError()) + ")";
                continue;
            }
            auto query =
                reinterpret_cast<int (*)(void (*)(const char*, const char*), char*, size_t)>(
                    GetProcAddress(library, "phoenix_backend_devices"));
            char error[2048]{};
            entries = Json::array();
            if (!query || query(device, error, sizeof(error)) < 0)
                report["errors"][kind] = query ? error : "Query ABI unavailable";
            else
                for (const auto& entry : entries)
                    report["devices"].push_back(entry);
            FreeLibrary(library);
        }
        auto vulkan_library =
            LoadLibraryExW((root / "vulkan/phoenix_vulkan.dll").c_str(), nullptr,
                           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (vulkan_library) {
            auto count =
                reinterpret_cast<int (*)()>(GetProcAddress(vulkan_library, "phoenix_device_count"));
            auto info = reinterpret_cast<int (*)(unsigned, char*, char*, unsigned*, unsigned*)>(
                GetProcAddress(vulkan_library, "phoenix_device_info"));
            if (count && info)
                for (int iteration_index = 0, n = count(); iteration_index < n; ++iteration_index) {
                    char device_id[33]{}, name[256]{};
                    unsigned vendor = 0, capable = 0;
                    if (!info(iteration_index, device_id, name, &vendor, &capable))
                        report["devices"].push_back({{"id", std::string("vulkan:") + device_id},
                                                     {"label", name},
                                                     {"vendor", vendor},
                                                     {"capable", bool(capable)}});
                }
            FreeLibrary(vulkan_library);
        } else
            report["errors"]["vulkan"] = "Provider unavailable";
        std::cout << report.dump(2) << '\n';
        if (cookie)
            RemoveDllDirectory(cookie);
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
