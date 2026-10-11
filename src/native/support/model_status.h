#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>

namespace phoenix_addon::phoenix {
enum class ModelDllStatus { unconfigured, missing, not_file, unreadable, available };

// UI preflight only. The loader still authenticates the DLL and its model
// resource before inference. Never load a module or hash model bytes here.
inline ModelDllStatus inspect_model_dll(const std::filesystem::path& path) {
    if (path.empty())
        return ModelDllStatus::unconfigured;
    std::error_code error;
    const auto status = std::filesystem::status(path, error);
    if (error == std::errc::no_such_file_or_directory)
        return ModelDllStatus::missing;
    if (error)
        return ModelDllStatus::unreadable;
    if (status.type() == std::filesystem::file_type::not_found)
        return ModelDllStatus::missing;
    if (!std::filesystem::is_regular_file(status))
        return ModelDllStatus::not_file;
    std::ifstream file(path, std::ios::binary);
    return file.is_open() ? ModelDllStatus::available : ModelDllStatus::unreadable;
}
inline const char* model_dll_warning(ModelDllStatus status) {
    switch (status) {
    case ModelDllStatus::unconfigured:
        return "DLSS5 Phoenix model DLL is not configured.";
    case ModelDllStatus::missing:
        return "DLSS5 Phoenix cannot find the NVIDIA model DLL.";
    case ModelDllStatus::not_file:
        return "DLSS5 Phoenix model DLL path points to something other than a file.";
    case ModelDllStatus::unreadable:
        return "DLSS5 Phoenix cannot access the NVIDIA model DLL.";
    default:
        return "";
    }
}
struct ModelDllMonitor {
    std::filesystem::path path;
    ModelDllStatus status = ModelDllStatus::unconfigured;
    uint64_t checked_at = 0;
    bool checked = false;
    ModelDllStatus poll(const std::filesystem::path& requested, uint64_t now) {
        if (!checked || path != requested || now < checked_at || now - checked_at >= 1000) {
            path = requested;
            status = inspect_model_dll(path);
            checked_at = now;
            checked = true;
        }
        return status;
    }
};
} // namespace production_game41::phoenix
