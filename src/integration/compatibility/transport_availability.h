#pragma once
#include "../../native/support/model_tools.h"
#include "../../native/model/bundle_loader.h"
#include <fstream>
#include <mutex>
#include <thread>
#include <algorithm>
#include <memory>
#include <vector>

namespace phoenix_compat {
inline std::filesystem::path transport_runtime_path(const std::filesystem::path& addon_folder,
        const std::string& configured, const std::filesystem::path& fallback) {
    auto path = configured.empty() ? fallback : std::filesystem::u8path(configured);
    if (path.has_root_name() && !path.is_absolute())
        throw std::runtime_error("Runtime path must be absolute or relative to the addon folder, not drive-relative.");
    if (path.is_relative()) path = addon_folder / path;
    return std::filesystem::absolute(path).lexically_normal();
}
struct TransportRuntimes {
    bool pending = true, native = false, portable = false;
    std::string native_reason = "Checking native runtime...";
    std::string portable_reason = "Checking portable runtime...";
    bool driver_graph = false;
    std::string driver_reason;
    std::vector<std::string> native_models{"original"};
};

inline bool transport_dll_matches(const std::filesystem::path& path, unsigned bits) {
    // Read PE headers as data. Availability checks must never load a GPU driver
    // or an optional runtime on the presentation thread.
    std::ifstream file(path, std::ios::binary);
    IMAGE_DOS_HEADER dos{};
    file.read(reinterpret_cast<char*>(&dos), sizeof(dos));
    if (!file || dos.e_magic != IMAGE_DOS_SIGNATURE || dos.e_lfanew < sizeof(dos) || dos.e_lfanew > 1024 * 1024)
        return false;
    file.seekg(dos.e_lfanew);
    DWORD signature{}; IMAGE_FILE_HEADER header{};
    file.read(reinterpret_cast<char*>(&signature), sizeof(signature));
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    return file && signature == IMAGE_NT_SIGNATURE && (header.Characteristics & IMAGE_FILE_DLL) &&
        header.Machine == (bits == 32 ? IMAGE_FILE_MACHINE_I386 : IMAGE_FILE_MACHINE_AMD64);
}

inline bool valid_transport_file_name(const std::string& name) {
    return !name.empty() && name != "." && name != ".." &&
        std::all_of(name.begin(), name.end(), [](unsigned char ch) { return ch >= 32 && ch != 127; }) &&
        name.find_first_of("/\\:*?\"<>|") == std::string::npos;
}

inline void inspect_native_transport(const std::filesystem::path& root,
                                     const std::filesystem::path& native_system) {
    using phoenix_bundle::require;
    const auto profile = phoenix_bundle::json_file(root / "bootstrap.json");
    require(profile.at("schema") == "phoenix-wow64-bootstrap/2", "Native bootstrap package needs an update.");
    const auto& files = profile.at("files");
    for (const char* name : {"wow64ext.dll", "phoenix_bootstrap64.dll", "phoenix_driver64.dll",
                            "phoenix-nvngx.dll", "phoenix_native.dll"})
        require(files.contains(name), "Native bootstrap package is incomplete.");
    for (const auto& item : files.items()) {
        require(valid_transport_file_name(item.key()) && item.value().is_string(), "Invalid native package member.");
        require(phoenix_embedded::file_digest(root / std::filesystem::u8path(item.key())) == item.value().get<std::string>(),
                "Native runtime package hash mismatch.");
    }
    require(transport_dll_matches(root / "wow64ext.dll", 32), "Native bridge must be an x86 DLL.");
    for (const char* name : {"phoenix_bootstrap64.dll", "phoenix_driver64.dll", "phoenix-nvngx.dll", "phoenix_native.dll"})
        require(transport_dll_matches(root / name, 64), "Native worker must be an x64 DLL.");
    require(std::filesystem::is_regular_file(root / "native/native_manifest.json"), "Native kernel bundle is missing.");
    require(phoenix_embedded::file_digest(native_system / "ntdll.dll") == profile.at("ntdll_sha256").get<std::string>(),
            "Native runtime profile does not match this Windows build.");
    const auto driver = profile.at("driver_directory").get<std::string>();
    require(valid_transport_file_name(driver), "Invalid native driver directory.");
    require(phoenix_embedded::file_digest(native_system / "DriverStore/FileRepository" /
                std::filesystem::u8path(driver) / "nvcuda64.dll") == profile.at("cuda_sha256").get<std::string>(),
            "Native runtime profile does not match this NVIDIA driver.");
}

inline void inspect_driver_transport_graph(const std::filesystem::path& root) {
    const auto manifest = phoenix_bundle::json_file(root / "native/native_manifest.json");
    const bool aligned = manifest.contains("native_graph_v2_sha256");
    const auto graph = root / "native" / (aligned ? "native_graph_v2.json" : "native_graph.json");
    phoenix_bundle::require(phoenix_embedded::file_digest(graph) ==
        manifest.at(aligned ? "native_graph_v2_sha256" : "native_graph_sha256").get<std::string>(),
        "Native driver graph hash mismatch.");
    const auto metadata = phoenix_bundle::json_file(graph);
    phoenix_bundle::require(metadata.at("schema") == "phoenix-native-graph/1", "Native driver graph schema mismatch.");
    phoenix_generated::validate_tile_layout(metadata);
}

inline TransportRuntimes inspect_transport_runtimes(const std::filesystem::path& native_root,
        const std::filesystem::path& portable_root, const std::string& portable_device, unsigned bits,
        bool allow_unvalidated = false) {
    TransportRuntimes result; result.pending = false;
    if (bits == 32) {
        try {
            wchar_t windows[MAX_PATH]{};
            phoenix_bundle::require(GetWindowsDirectoryW(windows, MAX_PATH) != 0, "Windows path unavailable.");
            inspect_native_transport(native_root, std::filesystem::path(windows) / "Sysnative");
            result.native = true; result.native_reason.clear();
            result.native_models = phoenix_bundle::json_file(native_root / "bootstrap.json")
                .value("supported_models", std::vector<std::string>{"original"});
            try {
                inspect_driver_transport_graph(native_root);
                result.driver_graph = true;
            } catch (const std::exception& error) { result.driver_reason = error.what(); }
        } catch (const std::exception& error) { result.native_reason = error.what(); }
    } else result.native_reason = "Native cross-bitness routes require an x86 game.";
    try {
        phoenix_bundle::require(transport_dll_matches(portable_root / "phoenix_runtime_cpp.dll", bits),
                                "Matching-bitness portable runtime is not installed.");
        const auto backend = portable_device == "auto" ? "vulkan" : portable_device.substr(0, portable_device.find(':'));
        phoenix_bundle::require(backend == "cpu" || backend == "vulkan" || backend == "cuda" || backend == "hip" || backend == "sycl",
                                "Choose a supported portable device.");
        phoenix_bundle::require(transport_dll_matches(portable_root / ("phoenix_" + backend + "_cpp.dll"), bits),
                                "Matching portable backend library is not installed.");
        if (backend != "cpu") {
            phoenix_bundle::require(transport_dll_matches(portable_root / backend / ("phoenix_" + backend + ".dll"), bits),
                                    "Matching portable device provider is not installed.");
        }
        nlohmann::json config = nlohmann::json::object();
        if (std::filesystem::is_regular_file(portable_root / "phoenix_native_cpp.json"))
            config = phoenix_bundle::json_file(portable_root / "phoenix_native_cpp.json");
        const auto devices = config.value("validated_devices", std::vector<std::string>{});
        const bool validated = portable_device == "auto"
            ? std::any_of(devices.begin(), devices.end(), [](const auto& id) { return id.starts_with("vulkan:"); })
            : std::find(devices.begin(), devices.end(), portable_device) != devices.end();
        phoenix_bundle::require(validated || allow_unvalidated || config.value("allow_unvalidated", false),
                                "Portable device has no validation entry; explicit opt-in is required.");
        result.portable = true; result.portable_reason.clear();
    } catch (const std::exception& error) { result.portable_reason = error.what(); }
    return result;
}

class TransportAvailabilityProbe {
    struct State { std::mutex mutex; TransportRuntimes result; };
    std::shared_ptr<State> state_ = std::make_shared<State>();
    std::string key_;
    uint64_t changed_at_ = 0;
    bool started_ = false;
  public:
    void invalidate() { key_.clear(); started_ = false; }
    TransportRuntimes poll(const std::filesystem::path& native_root,
                          const std::filesystem::path& portable_root, const std::string& device, bool allow_unvalidated) {
        const auto key = native_root.u8string() + u8"\n" + portable_root.u8string() + u8"\n" +
                         std::filesystem::u8path(device).u8string();
        const auto text = std::string(reinterpret_cast<const char*>(key.data()), key.size()) + (allow_unvalidated ? "\nyes" : "\nno");
        if (text != key_) {
            key_ = text; changed_at_ = GetTickCount64(); started_ = false;
            state_ = std::make_shared<State>();
        }
        // Debounce folder edits; hashing must not spawn a worker per keystroke.
        if (!started_ && GetTickCount64() - changed_at_ >= 250) {
            started_ = true;
            HMODULE pin{};
            if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                    reinterpret_cast<LPCWSTR>(&inspect_transport_runtimes), &pin)) {
                try {
                    std::thread([state = state_, native_root, portable_root, device, allow_unvalidated] {
                        auto result = inspect_transport_runtimes(native_root, portable_root, device, sizeof(void*) * 8, allow_unvalidated);
                        std::lock_guard lock(state->mutex); state->result = std::move(result);
                    }).detach();
                } catch (const std::exception& error) {
                    std::lock_guard lock(state_->mutex);
                    state_->result = {false, false, false, error.what(), error.what()};
                }
            } else {
                std::lock_guard lock(state_->mutex);
                state_->result = {false, false, false, "Cannot start runtime checks.", "Cannot start runtime checks."};
            }
        }
        std::lock_guard lock(state_->mutex); return state_->result;
    }
};
} // namespace phoenix_compat
