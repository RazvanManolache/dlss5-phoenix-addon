#include "../../src/integration/compatibility/transport_availability.h"
#include <iostream>

int main() {
    using namespace phoenix_compat;
    namespace fs = std::filesystem;
    unsigned checks = 0;
    auto verify = [&](bool condition) { ++checks; if (!condition) throw std::runtime_error("Transport availability contract failed"); };
    auto rejected = [&](auto work) {
        bool failed = false; try { work(); } catch (const std::exception&) { failed = true; }
        verify(failed);
    };
    const auto root = fs::temp_directory_path() / ("phoenix-transport-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(GetTickCount64()));
    fs::create_directories(root);
    struct Cleanup { fs::path root; ~Cleanup() { std::error_code error; fs::remove_all(root, error); } } cleanup{root};
    auto write = [](const fs::path& path, const std::string& text) {
        fs::create_directories(path.parent_path()); std::ofstream file(path, std::ios::binary); file << text;
    };
    auto dll = [](const fs::path& path, unsigned bits) {
        fs::create_directories(path.parent_path()); std::ofstream file(path, std::ios::binary);
        IMAGE_DOS_HEADER dos{}; dos.e_magic = IMAGE_DOS_SIGNATURE; dos.e_lfanew = sizeof(dos);
        IMAGE_FILE_HEADER header{}; header.Machine = bits == 32 ? IMAGE_FILE_MACHINE_I386 : IMAGE_FILE_MACHINE_AMD64;
        header.Characteristics = IMAGE_FILE_DLL;
        const DWORD signature = IMAGE_NT_SIGNATURE;
        file.write(reinterpret_cast<const char*>(&dos), sizeof(dos));
        file.write(reinterpret_cast<const char*>(&signature), sizeof(signature));
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    };
    const auto addon = root / "game/addons";
    const auto native = root / L"External runtimes \u03a9";
    const auto system = root / "fake-native-system";
    const auto portable = root / "Shared portable";
    verify(transport_runtime_path(addon, phoenix_model::source_path_string(native), "unused") == native);
    verify(transport_runtime_path(addon, "../../Shared portable", "unused") == portable);
    verify(transport_runtime_path(addon, "", "PhoenixRuntime/win-x86") == addon / "PhoenixRuntime/win-x86");
    rejected([&] { transport_runtime_path(addon, "C:relative", "unused"); });
    write(system / "ntdll.dll", "Synthetic Windows profile");
    write(system / "DriverStore/FileRepository/synthetic-driver/nvcuda64.dll", "Synthetic driver profile");
    nlohmann::json profile{{"schema", "phoenix-wow64-bootstrap/3"}, {"files", nlohmann::json::object()},
        {"driver_directory", "synthetic-driver"}, {"ntdll_sha256", phoenix_embedded::file_digest(system / "ntdll.dll")},
        {"cuda_sha256", phoenix_embedded::file_digest(system / "DriverStore/FileRepository/synthetic-driver/nvcuda64.dll")}};
    for (const char* name : {"wow64ext.dll", "phoenix_bootstrap64.dll", "phoenix_driver64.dll", "phoenix-nvngx.dll", "phoenix_native.dll"}) {
        dll(native / name, std::string(name) == "wow64ext.dll" ? 32 : 64);
        profile["files"][name] = phoenix_embedded::file_digest(native / name);
    }
    write(native / "native/native_manifest.json", "{}");
    auto save = [&] { write(native / "bootstrap.json", profile.dump()); };
    save(); inspect_native_transport(native, system); verify(true);
    profile["schema"]="phoenix-wow64-bootstrap/2";save();
    rejected([&] { inspect_native_transport(native, system); });
    profile["schema"]="phoenix-wow64-bootstrap/3";save();
    write(native / "phoenix_native.dll", "Tampered member");
    rejected([&] { inspect_native_transport(native, system); });
    dll(native / "phoenix_native.dll", 64);
    write(system / "ntdll.dll", "Different Windows build");
    rejected([&] { inspect_native_transport(native, system); });
    write(system / "ntdll.dll", "Synthetic Windows profile");
    write(system / "DriverStore/FileRepository/synthetic-driver/nvcuda64.dll", "Changed driver");
    rejected([&] { inspect_native_transport(native, system); });
    write(system / "DriverStore/FileRepository/synthetic-driver/nvcuda64.dll", "Synthetic driver profile");
    profile["files"]["../escape.dll"] = std::string(64, '0'); save();
    rejected([&] { inspect_native_transport(native, system); });
    profile["files"].erase("../escape.dll"); save();
    const auto layout = phoenix_model::native_layout("phoenix-network37/1");
    nlohmann::json graph{{"schema", "phoenix-native-graph/1"}, {"layers", nlohmann::json::array()},
        {"tensors", {{"tensor:023.output_downsample_optional", {{"producer", 0}, {"zero_padding", true}}}}}};
    for (unsigned index = 0; index < layout.launches(); ++index) graph["layers"].push_back({{"grid", {"T32"}}});
    write(native / "native/native_graph_v2.json", graph.dump());
    nlohmann::json graph_manifest{{"native_graph_v2_sha256", phoenix_embedded::file_digest(native / "native/native_graph_v2.json")}};
    write(native / "native/native_manifest.json", graph_manifest.dump());
    inspect_driver_transport_graph(native); verify(true);
    graph["layers"][layout.repack_in()]["grid"][0] = "T";
    write(native / "native/native_graph_v2.json", graph.dump());
    graph_manifest["native_graph_v2_sha256"] = phoenix_embedded::file_digest(native / "native/native_graph_v2.json");
    write(native / "native/native_manifest.json", graph_manifest.dump());
    rejected([&] { inspect_driver_transport_graph(native); });
    dll(native / "wow64ext.dll", 64); profile["files"]["wow64ext.dll"] = phoenix_embedded::file_digest(native / "wow64ext.dll"); save();
    rejected([&] { inspect_native_transport(native, system); });
    dll(portable / "phoenix_runtime_cpp.dll", 64); dll(portable / "phoenix_cpu_cpp.dll", 64);
    write(portable / "phoenix_native_cpp.json", R"({"validated_devices":["cpu"]})");
    verify(inspect_transport_runtimes({}, portable, "cpu", 64).portable);
    verify(!inspect_transport_runtimes({}, portable, "cpu", 32).portable);
    verify(!inspect_transport_runtimes({}, portable, "auto", 64).portable);
    dll(portable / "phoenix_vulkan_cpp.dll", 64); dll(portable / "vulkan/phoenix_vulkan.dll", 64);
    verify(!inspect_transport_runtimes({}, portable, "auto", 64).portable);
    verify(inspect_transport_runtimes({}, portable, "auto", 64, true).portable);
    write(native / "chosen-model.dll", "Synthetic path-only model fixture");
    verify(phoenix_model::find_nvidia_dll("chosen-model.dll", addon, native) == native / "chosen-model.dll");
    verify(phoenix_model::find_nvidia_dll(native / "chosen-model.dll", addon) == native / "chosen-model.dll");
    std::cout << "{\"passed\":true,\"cpu_only\":true,\"checks\":" << checks << "}\n";
}
