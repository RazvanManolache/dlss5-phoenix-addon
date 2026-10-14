// Minimal standalone client of the public C ABI. No ReShade, Python or graphics API required.
#include "../native_runtime/platform.h"
#include "../third_party/nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
using Json = nlohmann::json;
using namespace phoenix_cpp;
std::vector<float> plane(const Json &job, const char *name, size_t count, bool required) {
    if (!job.contains(name)) {
        if (required)
            throw std::runtime_error(std::string("Missing plane: ") + name);
        return {};
    }
    std::ifstream file(std::filesystem::u8path(job.at(name).get<std::string>()),
                       std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(count * sizeof(float)))
        throw std::runtime_error(std::string("Plane byte count differs: ") + name);
    std::vector<float> values(count);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(values.data()), std::streamsize(count * sizeof(float))))
        throw std::runtime_error("Plane read failed");
    return values;
}
int main(int argc, char **argv) {
    try {
        if (argc != 3)
            throw std::runtime_error(
                "Usage: phoenix_frame JOB.json NEW_OUTPUT.rgb32f | --devices RUNTIME_FOLDER");
        const bool vulkan_devices = std::string(argv[1]) == "--vulkan-devices";
        const bool devices = std::string(argv[1]) == "--devices" || vulkan_devices;
        Json job;
        if (!devices) {
            std::ifstream input(std::filesystem::u8path(argv[1]));
            input >> job;
        }
        const auto root = std::filesystem::absolute(std::filesystem::u8path(
            devices ? std::string(argv[2])
                    : job.at("config").at("runtime_root").get<std::string>()));
        const auto module =
            open_library(vulkan_devices ? root / "vulkan" / library_name("phoenix_vulkan")
                                        : root / library_name(devices ? "phoenix_metal_cpp"
                                                                      : "phoenix_runtime_cpp"));
        if (!module)
            throw std::runtime_error("Standalone library unavailable in runtime_root");
        auto library =
            std::unique_ptr<std::remove_pointer_t<LibraryHandle>, decltype(&close_library)>(
                module, close_library);
        if (vulkan_devices) {
            auto count =
                reinterpret_cast<int (*)()>(library_symbol(module, "phoenix_device_count"));
            auto info = reinterpret_cast<int (*)(unsigned, char *, char *, unsigned *, unsigned *)>(
                library_symbol(module, "phoenix_device_info"));
            if (!count || !info)
                throw std::runtime_error("Vulkan enumeration unavailable");
            const auto size = count();
            if (size < 0)
                throw std::runtime_error("Vulkan 1.3 loader/instance unavailable");
            Json list = Json::array();
            for (int index = 0; index < size; ++index) {
                char uuid[33]{}, name[256]{};
                unsigned vendor = 0, supported = 0;
                if (info(unsigned(index), uuid, name, &vendor, &supported))
                    throw std::runtime_error("Vulkan device query failed");
                list.push_back({{"id", "vulkan:" + std::string(uuid)},
                                {"name", name},
                                {"vendor", vendor},
                                {"supported", bool(supported)},
                                {"inference_validated", false}});
            }
            std::cout << list.dump() << '\n';
            return 0;
        }
        if (devices) {
            auto list = reinterpret_cast<size_t (*)(char *, size_t)>(
                library_symbol(module, "phoenix_metal_devices"));
            if (!list)
                throw std::runtime_error("Metal enumeration unavailable");
            std::vector<char> text(list(nullptr, 0));
            list(text.data(), text.size());
            std::cout << text.data() << '\n';
            return 0;
        }
        auto create = reinterpret_cast<void *(*)(const char *, unsigned, unsigned)>(
            library_symbol(module, "phoenix_runtime_create"));
        auto destroy =
            reinterpret_cast<void (*)(void *)>(library_symbol(module, "phoenix_runtime_destroy"));
        auto frame = reinterpret_cast<int (*)(void *, const float *, const float *, const float *,
                                              const float *, const char *, float *)>(
            library_symbol(module, "phoenix_runtime_frame"));
        auto error =
            reinterpret_cast<const char *(*)()>(library_symbol(module, "phoenix_runtime_error"));
        auto status = reinterpret_cast<size_t (*)(void *, char *, size_t)>(
            library_symbol(module, "phoenix_runtime_status"));
        if (!create || !destroy || !frame || !error || !status)
            throw std::runtime_error("Incomplete Phoenix C API");
        const unsigned width = job.at("width"), height = job.at("height");
        if (!width || !height || width > 8192 || height > 8192)
            throw std::runtime_error("Frame extent must be 1..8192");
        const size_t pixels = size_t(width) * height;
        const auto output_path = std::filesystem::u8path(argv[2]);
        if (std::filesystem::exists(output_path))
            throw std::runtime_error("Fresh output path required");
        auto color = plane(job, "color", pixels * 3, true),
             motion = plane(job, "motion", pixels * 2, false),
             depth = plane(job, "depth", pixels, false),
             validity = plane(job, "validity", pixels, false);
        std::vector<float> output(pixels * 3);
        auto config = job.at("config").dump();
        auto session = std::unique_ptr<void, decltype(destroy)>(
            create(config.c_str(), width, height), destroy);
        if (!session)
            throw std::runtime_error(error());
        auto metadata = job.value("metadata", Json::object());
        metadata["has_motion"] = !motion.empty();
        metadata["has_depth"] = !depth.empty();
        metadata["has_history_validity"] = !validity.empty();
        metadata["reset"] = true;
        const auto text = metadata.dump();
        if (frame(session.get(), color.data(), motion.empty() ? nullptr : motion.data(),
                  depth.empty() ? nullptr : depth.data(),
                  validity.empty() ? nullptr : validity.data(), text.c_str(), output.data()))
            throw std::runtime_error(error());
        std::ofstream file(output_path, std::ios::binary);
        if (!file.write(reinterpret_cast<const char *>(output.data()),
                        std::streamsize(output.size() * sizeof(float))))
            throw std::runtime_error("Output write failed");
        std::vector<char> report(status(session.get(), nullptr, 0));
        status(session.get(), report.data(), report.size());
        std::cout << report.data() << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
