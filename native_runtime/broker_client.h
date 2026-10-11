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
        if (job.value)
            TerminateJobObject(job, 0);
        if (header)
            UnmapViewOfFile(header);
    }
    void open(const std::filesystem::path& root, const std::string& config, unsigned width,
              unsigned height, HANDLE cancellation = nullptr) {
        cancel = cancellation;
        pixels = size_t(width) * height;
        auto bytes = broker_bytes(width, height);
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
              const std::string& metadata, float* output) {
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

  public:
    void set_cancellation(HANDLE handle) {
        cancellation = handle;
    }
    void open(const std::filesystem::path& root, const nlohmann::json& config, unsigned width,
              unsigned height) {
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
    }
    int frame(const float* color, const float* motion, const float* depth, const float* validity,
              const std::string& metadata, float* output) {
        int result = broker ? broker->frame(color, motion, depth, validity, metadata, output)
                            : local.frame(local.instance, color, motion, depth, validity,
                                          metadata.c_str(), output);
        if (result)
            error_text = broker ? broker->status_json() : local.error();
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
