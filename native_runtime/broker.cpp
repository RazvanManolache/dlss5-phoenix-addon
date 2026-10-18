// Isolated process that executes requests from the addon over shared memory/events.
// The wire header and sequence handshake keep host pointers out of the process boundary.

#include "broker_protocol.h"
#include "runtime_api.h"
#include "thread_config.h"
#include "../third_party/nlohmann/json.hpp"
#include <memory>

// Never run by the build. The addon launches this optional x64 helper hidden.
int wmain(int argc, wchar_t** argv) {
    using namespace phoenix_cpp;
    if (argc != 3)
        return 2;
    BrokerHeader* header = nullptr;
    Handle mapping, request, complete, stop, parent;
    try {
        const std::wstring name = argv[1];
        if (name.rfind(L"Local\\PhoenixCpp_", 0) != 0 || name.size() > 150)
            return 3;
        const unsigned long pid = std::stoul(argv[2]);
        parent.value = require_handle(OpenProcess(SYNCHRONIZE, FALSE, pid));
        mapping.value = require_handle(OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name.c_str()));
        auto prefix = static_cast<BrokerHeader*>(
            MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(BrokerHeader)));
        if (!prefix)
            throw std::runtime_error("Broker header mapping failed");
        auto bytes = prefix->bytes;
        auto width = prefix->width, height = prefix->height;
        const auto protocol = prefix->version;
        bool valid = prefix->magic == broker_magic && (protocol == broker_version || protocol == broker_extended_version) &&
                     prefix->parent == pid;
        UnmapViewOfFile(prefix);
        if (!valid || bytes != broker_bytes(width, height, protocol))
            throw std::runtime_error("Broker protocol mismatch");
        header = static_cast<BrokerHeader*>(
            MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, size_t(bytes)));
        if (!header)
            throw std::runtime_error("Broker plane mapping failed");
        request.value =
            require_handle(OpenEventW(SYNCHRONIZE, FALSE, (name + L"_request").c_str()));
        complete.value =
            require_handle(OpenEventW(EVENT_MODIFY_STATE, FALSE, (name + L"_complete").c_str()));
        stop.value = require_handle(OpenEventW(SYNCHRONIZE, FALSE, (name + L"_stop").c_str()));
        if (!std::memchr(header->config, 0, text_capacity))
            throw std::runtime_error("Broker config not terminated");
        auto config = nlohmann::json::parse(header->config);
        if (config.value("device", std::string()) == "sycl:cpu") {
            const auto count =
                std::to_wstring(phoenix_threads::resolve(cpu_thread_request(config)));
            if (!SetEnvironmentVariableW(L"CL_CONFIG_CPU_TBB_NUM_WORKERS", count.c_str()))
                throw std::runtime_error("CPU worker count setup failed");
        }
        // A private OpenCL ICD is process-local to this helper. Never register
        // it globally or change the game's environment/search directories.
        struct DllDirectory {
            DLL_DIRECTORY_COOKIE cookie = nullptr;
            ~DllDirectory() {
                if (cookie)
                    RemoveDllDirectory(cookie);
            }
        } directory;
        struct CpuOpenCL {
            HMODULE module = nullptr;
            ~CpuOpenCL() {
                if (module)
                    FreeLibrary(module);
            }
        } cpu_opencl;
        if (config.value("device", std::string()) == "sycl:cpu" &&
            config.contains("private_opencl_cpu")) {
            auto root = std::filesystem::u8path(config.at("runtime_root").get<std::string>());
            auto relative =
                std::filesystem::u8path(config.at("private_opencl_cpu").get<std::string>());
            if (relative.is_absolute() || relative.has_root_path())
                throw std::runtime_error("Private CPU ICD must be package-relative");
            for (const auto& part : relative)
                if (part == L"..")
                    throw std::runtime_error("Private CPU ICD leaves package");
            auto icd = root / relative;
            if (!std::filesystem::is_regular_file(icd))
                throw std::runtime_error("Private CPU ICD missing");
            if (_wcsicmp(icd.filename().c_str(), L"intelocl64.dll") != 0)
                throw std::runtime_error("Private CPU forwarding library requires intelocl64.dll");
            directory.cookie = AddDllDirectory(icd.parent_path().c_str());
            if (!directory.cookie)
                throw std::runtime_error("Private CPU ICD setup failed");
            // OCL_ICD_FILENAMES adds an ICD but still enumerates installed GPU
            // drivers. Bind OpenCL imports to the CPU ICD in this owned helper
            // instead, before the SYCL adapter is loaded. The game is untouched.
            cpu_opencl.module =
                LoadLibraryExW((icd.parent_path() / L"OpenCL.dll").c_str(), nullptr,
                               LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if (!cpu_opencl.module)
                throw std::runtime_error(
                    "Private CPU OpenCL forwarding library missing; build opencl_cpu_proxy");
        }
        RuntimeApi api;
        api.open(std::filesystem::u8path(config.at("runtime_root").get<std::string>()),
                 header->config, width, height);
        header->ready = 1;
        broker_text(header->status, api.status_json());
        SetEvent(complete);
        const size_t pixel_count = size_t(width) * height;
        auto color = reinterpret_cast<float*>(header + 1);
        auto motion = color + 3 * pixel_count, depth = motion + 2 * pixel_count,
             validity = depth + pixel_count, output = validity + pixel_count;
        const HANDLE handles[] = {parent, stop, request};
        uint64_t sequence = 0;
        while (WaitForMultipleObjects(3, handles, FALSE, INFINITE) == WAIT_OBJECT_0 + 2) {
            if (!std::memchr(header->metadata, 0, text_capacity) || header->sequence <= sequence)
                throw std::runtime_error("Broker request sequence invalid");
            sequence = header->sequence;
            if (header->version != protocol || header->bytes != bytes || header->width != width ||
                header->height != height || (header->reserved & ~31u) ||
                (protocol == broker_version && header->reserved))
                throw std::runtime_error("Broker request layout changed");
            if (!header->reserved)
                header->result = api.frame(api.instance, color, motion, depth, validity, header->metadata, output);
            else if (!api.frame_v3) {
                header->result = -1;
                broker_text(header->status, "{\"ok\":false,\"error\":\"Runtime lacks advanced frame API\"}");
                SetEvent(complete);
                continue;
            } else {
                const auto metadata = nlohmann::json::parse(header->metadata);
                auto view = [&](const float* data, unsigned channels, bool present = true) {
                    return present ? phoenix_plane_f32{data, pixel_count*channels*sizeof(float),0,width,height,channels,0}
                        : phoenix_plane_f32{};
                };
                phoenix_frame_v3 request{};
                request.base.struct_bytes=sizeof(request); request.base.version=PHOENIX_FRAME_VERSION_3;
                request.base.control_profile=(header->reserved & BrokerMask) ? PHOENIX_CONTROL_EXPERIMENTAL_RGB_1 : PHOENIX_CONTROL_NONE;
                request.base.color=view(color,3);
                request.base.motion=view(motion,2,metadata.value("has_motion",false));
                request.base.depth=view(depth,1,metadata.value("has_depth",false));
                request.base.validity=view(validity,1,metadata.value("has_history_validity",false));
                request.base.control_mask=view(color+10*pixel_count,4,header->reserved & BrokerMask);
                request.base.output={output,pixel_count*3*sizeof(float),0,width,height,3,0};
                request.base.metadata_json=header->metadata;
                request.ui=view(color+14*pixel_count,4,header->reserved & BrokerUI);
                request.ui_alpha=view(color+18*pixel_count,1,header->reserved & BrokerAlpha);
                request.backbuffer=view(color+19*pixel_count,3,header->reserved & BrokerBackbuffer);
                request.distortion=view(color+22*pixel_count,4,header->reserved & BrokerDistortion);
                header->result=api.frame_v3(api.instance,&request);
            }
            broker_text(header->status,
                        header->result
                            ? nlohmann::json({{"ok", false}, {"error", api.error()}}).dump()
                            : api.status_json());
            SetEvent(complete);
        }
        UnmapViewOfFile(header);
        return 0;
    } catch (const std::exception& exception) {
        if (header && complete.value) {
            header->result = -1;
            header->ready = 2;
            try {
                broker_text(header->status,
                            nlohmann::json({{"ok", false}, {"error", exception.what()}}).dump());
            } catch (...) {
            }
            SetEvent(complete);
        }
        if (header)
            UnmapViewOfFile(header);
        return 1;
    }
}
