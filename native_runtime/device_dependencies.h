#pragma once
#include <filesystem>
#include <string>
#include "../third_party/nlohmann/json.hpp"
namespace phoenix_addon::phoenix::portable {
inline bool device_dependencies_present(const std::filesystem::path& root,
                                        const nlohmann::json& config, const std::string& device) {
    if (device!="sycl:cpu") return true;
    // The broker requires this private CPU ICD; a provider DLL and an old
    // validation receipt alone do not make the packaged runner usable.
    const auto relative=std::filesystem::u8path(config.value("private_opencl_cpu",std::string()));
    if (relative.empty() || relative.is_absolute()) return false;
    for (const auto& component:relative) if (component=="..") return false;
    return std::filesystem::is_regular_file(root/relative);
}
}
