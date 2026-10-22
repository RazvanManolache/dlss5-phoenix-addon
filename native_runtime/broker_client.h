// Starts and owns the portable broker process and its shared-memory handshake.
// A request is complete only after the matching sequence reports completion or a bounded failure.

#pragma once
#include "broker_protocol.h"
#include "runtime_api.h"
#include "../third_party/nlohmann/json.hpp"
#include <bcrypt.h>
#include <algorithm>

namespace phoenix_cpp {
class BrokerClient {
    Handle mapping, request, complete, stop, job, process;
    BrokerHeader* header = nullptr;
    uint64_t sequence = 0;
    size_t pixels = 0;
    uint32_t protocol = broker_extended_version;
    HANDLE cancel = nullptr;
    void wait(DWORD timeout = 600000) {
        const HANDLE handles[] = {complete, process, cancel};
        auto result = WaitForMultipleObjects(cancel ? 3 : 2, handles, FALSE, timeout);
        if (result != WAIT_OBJECT_0)
            throw std::runtime_error(result == WAIT_TIMEOUT        ? "Native broker timed out"
                                     : result == WAIT_OBJECT_0 + 2 ? "Native broker cancelled"
                                                                   : "Native broker exited");
    }

  public:
    BrokerClient() = default;
    BrokerClient(const BrokerClient&) = delete;
    ~BrokerClient() {
        if (stop.value)
            SetEvent(stop);
        // Idle workers can release backend contexts normally. Terminating a job
        // is asynchronous: wait for process exit before its owner unloads DLLs
        // or removes an extracted package containing those mapped libraries.
        if (process.value && WaitForSingleObject(process, 5000) == WAIT_TIMEOUT) {
            if (job.value) TerminateJobObject(job, 0);
            WaitForSingleObject(process, 5000);
        }
        if (header)
            UnmapViewOfFile(header);
    }
    void open(const std::filesystem::path& root, const std::string& config, unsigned width,
              unsigned height, HANDLE cancellation = nullptr) {
        cancel = cancellation;
        pixels = size_t(width) * height;
        protocol = nlohmann::json::parse(config).value("broker_protocol", broker_extended_version);
        auto bytes = broker_bytes(width, height, protocol);
        if(bytes>SIZE_MAX)throw std::runtime_error("Broker mapping exceeds caller address space");
        unsigned char random[16];
        if (BCryptGenRandom(nullptr, random, sizeof(random), BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0)
            throw std::runtime_error("Broker nonce generation failed");
        const wchar_t hex[] = L"0123456789abcdef";
        std::wstring name = L"Local\\PhoenixCpp_";
        for (auto random_word : random) {
            name += hex[random_word >> 4];
            name += hex[random_word & 15];
        }
        mapping.value =
            require_handle(CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                              DWORD(bytes >> 32), DWORD(bytes), name.c_str()));
        if (GetLastError() == ERROR_ALREADY_EXISTS)
            throw std::runtime_error("Broker mapping collision");
        header = static_cast<BrokerHeader*>(
            MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, size_t(bytes)));
        if (!header)
            throw std::runtime_error("Broker memory allocation failed");
        new (header) BrokerHeader();
        header->version = protocol;
        header->width = width;
        header->height = height;
        header->bytes = bytes;
        header->parent = GetCurrentProcessId();
        broker_text(header->config, config);
        request.value =
            require_handle(CreateEventW(nullptr, FALSE, FALSE, (name + L"_request").c_str()));
        complete.value =
            require_handle(CreateEventW(nullptr, FALSE, FALSE, (name + L"_complete").c_str()));
        stop.value = require_handle(CreateEventW(nullptr, TRUE, FALSE, (name + L"_stop").c_str()));
        job.value = require_handle(CreateJobObjectW(nullptr, nullptr));
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits,
                                     sizeof(limits)))
            throw std::runtime_error("Broker job configuration failed");
        auto executable = root / L"phoenix_native_broker.exe";
        std::wstring command = L"\"" + executable.wstring() + L"\" " + name + L" " +
                               std::to_wstring(GetCurrentProcessId());
        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESHOWWINDOW;
        startup.wShowWindow = SW_HIDE;
        PROCESS_INFORMATION process_info{};
        if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, FALSE,
                            CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, root.c_str(), &startup,
                            &process_info))
            throw std::runtime_error("Native broker launch failed");
        process.value = process_info.hProcess;
        Handle thread(process_info.hThread);
        if (!AssignProcessToJobObject(job, process)) {
            TerminateProcess(process, 1);
            throw std::runtime_error("Native broker lifetime binding failed");
        }
        if (ResumeThread(thread) == DWORD(-1))
            throw std::runtime_error("Native broker resume failed");
        wait(120000);
        if (header->ready != 1)
            throw std::runtime_error(header->status);
    }
    int frame(const float* color, const float* motion, const float* depth, const float* validity,
              const std::string& metadata, float* output, const float* mask = nullptr,
              const float* ui = nullptr, const float* alpha = nullptr,
              const float* backbuffer = nullptr, const float* distortion = nullptr) {
        const uint32_t flags = (mask ? BrokerMask : 0) | (ui ? BrokerUI : 0) |
            (alpha ? BrokerAlpha : 0) | (backbuffer ? BrokerBackbuffer : 0) |
            (distortion ? BrokerDistortion : 0);
        if (flags && protocol != broker_extended_version)
            throw std::runtime_error("Advanced planes require broker protocol 2");
        auto base = reinterpret_cast<float*>(header + 1);
        std::memcpy(base, color, pixels * 3 * sizeof(float));
        auto copy = [&](size_t offset, size_t count, const float* src) {
            if (src)
                std::memcpy(base + offset, src, count * sizeof(float));
            else
                std::fill(base + offset, base + offset + count, 0.f);
        };
        copy(3 * pixels, 2 * pixels, motion);
        copy(5 * pixels, pixels, depth);
        copy(6 * pixels, pixels, validity);
        if (protocol == broker_extended_version) {
            copy(10*pixels, 4*pixels, mask);
            copy(14*pixels, 4*pixels, ui);
            copy(18*pixels, pixels, alpha);
            copy(19*pixels, 3*pixels, backbuffer);
            copy(22*pixels, 4*pixels, distortion);
        }
        header->reserved = flags;
        broker_text(header->metadata, metadata);
        header->sequence = ++sequence;
        SetEvent(request);
        wait();
        if (!header->result)
            std::memcpy(output, base + 7 * pixels, 3 * pixels * sizeof(float));
        return header->result;
    }
    DWORD process_id() const {
        return GetProcessId(process);
    }
    std::string status_json() const {
        if (!std::memchr(header->status, 0, text_capacity))
            throw std::runtime_error("Broker status not terminated");
        return header->status;
    }
};
class NativeSession {
    RuntimeApi local;
    std::unique_ptr<BrokerClient> broker;
    std::string error_text;
    HANDLE cancellation = nullptr;
    unsigned width_ = 0, height_ = 0;
    unsigned pipeline_contract_=0;
    bool accepts_pipeline(const std::string& metadata){
        if(pipeline_contract_>=1)return true;
        if(!nlohmann::json::parse(metadata).contains("pipeline"))return true;
        error_text="Update the portable runtime before using custom processing chains";return false;
    }

  public:
    void set_cancellation(HANDLE handle) {
        cancellation = handle;
    }
    void open(const std::filesystem::path& root, const nlohmann::json& config, unsigned width,
              unsigned height) {
        width_ = width; height_ = height;
        auto effective = config;
        if (config.value("device", std::string()) == "sycl:cpu")
            effective["execution"] = "broker";
        const auto mode = effective.value("execution", std::string("in_process"));
        if (mode == "broker") {
            broker = std::make_unique<BrokerClient>();
            broker->open(root, effective.dump(), width, height, cancellation);
        } else if (mode == "in_process")
            local.open(root, config.dump(), width, height);
        else
            throw std::runtime_error("Unknown native execution mode");
        pipeline_contract_=nlohmann::json::parse(status_json()).value("portable_pipeline_contract",0u);
    }
    int frame(const float* color, const float* motion, const float* depth, const float* validity,
              const std::string& metadata, float* output) {
        if(!accepts_pipeline(metadata))return -1;
        int result = broker ? broker->frame(color, motion, depth, validity, metadata, output)
                            : local.frame(local.instance, color, motion, depth, validity,
                                          metadata.c_str(), output);
        if (result)
            error_text = broker ? broker->status_json() : local.error();
        return result;
    }
    int frame_advanced(const float* color, const float* motion, const float* depth,
              const float* validity, const std::string& metadata, float* output,
              const float* mask = nullptr, const float* ui = nullptr, const float* alpha = nullptr,
              const float* backbuffer = nullptr, const float* distortion = nullptr) {
        if (!mask && !ui && !alpha && !backbuffer && !distortion)
            return frame(color, motion, depth, validity, metadata, output);
        if(!accepts_pipeline(metadata))return -1;
        int result = -1;
        if (broker) result = broker->frame(color, motion, depth, validity, metadata, output,
            mask, ui, alpha, backbuffer, distortion);
        else if (!local.frame_v3) error_text = "Advanced frame API unavailable in selected runtime";
        else {
            const size_t pixels = size_t(width_) * height_;
            auto view = [&](const float* data, unsigned channels) {
                return data ? phoenix_plane_f32{data, pixels * channels * sizeof(float), 0,
                    width_, height_, channels, 0} : phoenix_plane_f32{};
            };
            phoenix_frame_v3 request{};
            request.base.struct_bytes = sizeof(request);
            request.base.version = PHOENIX_FRAME_VERSION_3;
            request.base.control_profile = mask ? PHOENIX_CONTROL_EXPERIMENTAL_RGB_1 : PHOENIX_CONTROL_NONE;
            request.base.color=view(color,3); request.base.motion=view(motion,2);
            request.base.depth=view(depth,1); request.base.validity=view(validity,1);
            request.base.control_mask=view(mask,4);
            request.base.output={output,pixels*3*sizeof(float),0,width_,height_,3,0};
            request.base.metadata_json=metadata.c_str();
            request.ui=view(ui,4); request.ui_alpha=view(alpha,1);
            request.backbuffer=view(backbuffer,3); request.distortion=view(distortion,4);
            result=local.frame_v3(local.instance,&request);
            if (result) error_text=local.error();
        }
        if (result && broker) error_text=broker->status_json();
        return result;
    }
    DWORD process_id() const {
        return broker ? broker->process_id() : GetCurrentProcessId();
    }
    const char* error() const {
        return error_text.c_str();
    }
    std::string status_json() {
        return broker ? broker->status_json() : local.status_json();
    }
};
}
