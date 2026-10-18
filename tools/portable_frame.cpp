// Minimal standalone client of the public C ABI. No ReShade, Python or graphics API required.
#include "../native_runtime/platform.h"
#include "../native_runtime/phoenix.h"
#include "../third_party/nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#ifdef _WIN32
#include "../native_runtime/broker_client.h"
#endif
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
        bool broker_mode=false;
#ifdef _WIN32
        broker_mode=!devices&&job.at("config").value("execution",std::string("in_process"))=="broker";
        std::unique_ptr<NativeSession> broker_session;
#endif
        const auto module =
            broker_mode?nullptr:open_library(vulkan_devices ? root / "vulkan" / library_name("phoenix_vulkan")
                                        : root / library_name(devices ? "phoenix_metal_cpp"
                                                                      : "phoenix_runtime_cpp"));
        if (!module&&!broker_mode)
            throw std::runtime_error("Standalone library unavailable in runtime_root");
        auto library =
            std::unique_ptr<std::remove_pointer_t<LibraryHandle>, decltype(&close_library)>(
                module, close_library);
        if (vulkan_devices) {
            auto count =
                reinterpret_cast<int (*)()>(library_symbol(module, "phoenix_device_count"));
            auto info = reinterpret_cast<int (*)(unsigned, char *, char *, unsigned *, unsigned *)>(
                library_symbol(module, "phoenix_device_info"));
            auto optional = reinterpret_cast<int (*)(unsigned, unsigned, char *, char *, unsigned *, unsigned *)>(
                library_symbol(module, "phoenix_device_info_options"));
            if (!count || !info)
                throw std::runtime_error("Vulkan enumeration unavailable");
            const auto size = count();
            if (size < 0)
                throw std::runtime_error("Vulkan 1.3 loader/instance unavailable");
            Json list = Json::array();
            for (int index = 0; index < size; ++index) {
                char uuid[33]{}, name[256]{};
                unsigned vendor = 0, supported = 0, compatibility = 0;
                if (info(unsigned(index), uuid, name, &vendor, &supported))
                    throw std::runtime_error("Vulkan device query failed");
                if (optional && optional(unsigned(index), 1u << 30, uuid, name, &vendor, &compatibility))
                    throw std::runtime_error("Vulkan compatibility query failed");
                list.push_back({{"id", "vulkan:" + std::string(uuid)},
                                {"name", name},
                                {"vendor", vendor},
                                {"supported", bool(supported)},
                                {"compatibility_supported", bool(compatibility)},
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
        auto symbol=[&](const char* name){return module?library_symbol(module,name):nullptr;};
        auto create = reinterpret_cast<void *(*)(const char *, unsigned, unsigned)>(
            symbol("phoenix_runtime_create"));
        auto destroy =
            reinterpret_cast<void (*)(void *)>(symbol("phoenix_runtime_destroy"));
        auto frame = reinterpret_cast<int (*)(void *, const float *, const float *, const float *,
                                              const float *, const char *, float *)>(
            symbol("phoenix_runtime_frame"));
        auto frame_v2 = reinterpret_cast<int (*)(void*, const phoenix_frame_v2*)>(
            symbol("phoenix_runtime_frame_v2"));
        auto frame_v3 = reinterpret_cast<int (*)(void*, const phoenix_frame_v3*)>(
            symbol("phoenix_runtime_frame_v3"));
        auto error =
            reinterpret_cast<const char *(*)()>(symbol("phoenix_runtime_error"));
        auto status = reinterpret_cast<size_t (*)(void *, char *, size_t)>(
            symbol("phoenix_runtime_status"));
        if (!broker_mode&&(!create || !destroy || !frame || !error || !status))
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
             validity = plane(job, "validity", pixels, false),
             mask = plane(job, "control_mask", pixels * 4, false),
             ui = plane(job, "ui", pixels * 4, false), alpha = plane(job, "ui_alpha", pixels, false),
             backbuffer = plane(job, "backbuffer", pixels * 3, false),
             distortion = plane(job, "distortion", pixels * 4, false);
        if ((!ui.empty() || !alpha.empty() || !backbuffer.empty() || !distortion.empty()) && !frame_v3&&!broker_mode)
            throw std::runtime_error("Advanced planes require frame API V3");
        if (!mask.empty() && ((!frame_v2&&!broker_mode) ||
            job.value("control_profile", std::string()) != "phoenix-experimental-rgb/1"))
            throw std::runtime_error("Control mask requires V2 and explicit phoenix-experimental-rgb/1 profile");
        std::vector<float> output(pixels * 3);
        auto config = job.at("config").dump();
#ifdef _WIN32
        if(broker_mode) {broker_session=std::make_unique<NativeSession>();broker_session->open(root,job.at("config"),width,height);}
#endif
        auto session = std::unique_ptr<void, decltype(destroy)>(
            broker_mode?nullptr:create(config.c_str(), width, height), destroy);
        if (!session&&!broker_mode)
            throw std::runtime_error(error());
        auto metadata = job.value("metadata", Json::object());
        metadata["has_motion"] = !motion.empty();
        metadata["has_depth"] = !depth.empty();
        metadata["has_history_validity"] = !validity.empty();
        metadata["reset"] = true;
        const auto text = metadata.dump();
        phoenix_frame_v2 bounded{};
        bounded.struct_bytes = sizeof(bounded);
        bounded.version = PHOENIX_FRAME_VERSION_2;
        bounded.control_profile = mask.empty() ? PHOENIX_CONTROL_NONE : PHOENIX_CONTROL_EXPERIMENTAL_RGB_1;
        auto view = [&](const std::vector<float>& values, unsigned channels) {
            return values.empty() ? phoenix_plane_f32{} : phoenix_plane_f32{
                values.data(), values.size() * sizeof(float), 0, width, height, channels, 0};
        };
        bounded.color = view(color, 3);
        bounded.motion = view(motion, 2);
        bounded.depth = view(depth, 1);
        bounded.validity = view(validity, 1);
        bounded.control_mask = view(mask, 4);
        bounded.output = {output.data(), output.size() * sizeof(float), 0, width, height, 3, 0};
        bounded.metadata_json = text.c_str();
        phoenix_frame_v3 extended{};extended.base=bounded;
        extended.base.struct_bytes=sizeof(extended);extended.base.version=PHOENIX_FRAME_VERSION_3;
        extended.ui=view(ui,4);extended.ui_alpha=view(alpha,1);
        extended.backbuffer=view(backbuffer,3);extended.distortion=view(distortion,4);
        const unsigned iterations = job.value("benchmark_iterations", 1u);
        if (!iterations || iterations > 100)
            throw std::runtime_error("Benchmark iterations must be 1..100");
        std::vector<char> report;
        for (unsigned iteration = 0; iteration < iterations; ++iteration) {
            const auto started = std::chrono::steady_clock::now();
            int result=0;
#ifdef _WIN32
            if(broker_mode) result=broker_session->frame_advanced(color.data(),motion.empty()?nullptr:motion.data(),
                depth.empty()?nullptr:depth.data(),validity.empty()?nullptr:validity.data(),text,output.data(),
                mask.empty()?nullptr:mask.data(),ui.empty()?nullptr:ui.data(),alpha.empty()?nullptr:alpha.data(),
                backbuffer.empty()?nullptr:backbuffer.data(),distortion.empty()?nullptr:distortion.data());
            else
#endif
            result = frame_v3 ? frame_v3(session.get(), &extended) : frame_v2 ? frame_v2(session.get(), &bounded) :
                frame(session.get(), color.data(), motion.empty() ? nullptr : motion.data(),
                      depth.empty() ? nullptr : depth.data(),
                      validity.empty() ? nullptr : validity.data(), text.c_str(), output.data());
            if (result) {
#ifdef _WIN32
                if(broker_mode)throw std::runtime_error(broker_session->error());
#endif
                throw std::runtime_error(error());
            }
            const double call_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - started).count();
#ifdef _WIN32
            if(broker_mode){const auto value=broker_session->status_json();report.assign(value.begin(),value.end());report.push_back(0);}
            else
#endif
            {report.resize(status(session.get(), nullptr, 0));status(session.get(), report.data(), report.size());}
            if (iterations > 1) {
                auto row = Json::parse(report.data());
                row["benchmark_iteration"] = iteration;
                row["call_ms"] = call_ms;
                std::cout << row.dump() << std::endl;
            }
        }
        std::ofstream file(output_path, std::ios::binary);
        if (!file.write(reinterpret_cast<const char *>(output.data()),
                        std::streamsize(output.size() * sizeof(float))))
            throw std::runtime_error("Output write failed");
        if (iterations == 1)
            std::cout << report.data() << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
