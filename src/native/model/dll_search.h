#pragma once
#include <filesystem>
#include <stdexcept>
#include <string>

namespace phoenix_model {
// Search only explicit locations, never PATH/CWD or a recursive library tree.
// A local file wins even if incompatible: its identity check must fail visibly
// rather than silently substituting a different model/runtime edition.
inline std::filesystem::path find_nvidia_dll(const std::filesystem::path& requested,
                                            const std::filesystem::path& local_directory,
                                            const std::filesystem::path& search_folder = {}) {
    using Path = std::filesystem::path;
    auto validate = [](const Path& path) {
        if (path.native().find(Path::value_type(0)) != Path::string_type::npos ||
            (!path.empty() && path.has_root_name() && !path.is_absolute()))
            throw std::runtime_error("Invalid NVIDIA DLSS5 library path");
    };
    validate(requested);
    validate(search_folder);
    if (!local_directory.is_absolute())
        throw std::runtime_error("NVIDIA DLSS5 search needs an absolute local directory");
    const auto name = requested.empty() ? Path("nvngx_dlssnr.dll") : requested.filename();
    if (name.empty() || name == "." || name == "..")
        throw std::runtime_error("NVIDIA DLSS5 DLL filename is empty");
    const auto local = requested.empty() ? local_directory / name
                     : requested.is_absolute() ? requested : local_directory / requested;
    std::error_code error;
    if (std::filesystem::is_regular_file(local, error))
        return local.lexically_normal();
    if (!search_folder.empty()) {
        const auto folder = search_folder.is_absolute() ? search_folder : local_directory / search_folder;
        const auto shared = (folder / name).lexically_normal();
        error.clear();
        if (std::filesystem::is_regular_file(shared, error))
            return shared;
    }
    return {};
}
} // namespace phoenix_model
