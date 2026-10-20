// Connects addon frame resources to the native C++ portable executor.
// Model source and selected device belong to a runtime instance, not to a global implicit default.

#pragma once
#include <windows.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include "broker_client.h"
#include "device_dependencies.h"
#include "../src/native/support/cpu_threads.h"
#include "../src/native/model/weight_source.h"
#include "../src/native/support/device_selection.h"
namespace phoenix_addon::phoenix::portable {
using Json = nlohmann::json;
enum class WorkerState : uint32_t { idle, queued, running, result, error, stop };
inline void native_client_anchor() {}
inline std::filesystem::path native_client_root() {
    HMODULE module{};
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                            reinterpret_cast<LPCWSTR>(&native_client_anchor), &module))
        throw std::runtime_error("Addon module location unavailable");
    std::vector<wchar_t> path(32768);
    auto path_length = GetModuleFileNameW(module, path.data(), DWORD(path.size()));
    if (!path_length || path_length >= path.size())
        throw std::runtime_error("Addon module path truncated");
    return std::filesystem::path(std::wstring(path.data(), path_length)).parent_path();
}
inline std::filesystem::path& configured_runtime_root() {
    static std::filesystem::path path;
    return path;
}
inline std::filesystem::path native_runtime_root() {
    return configured_runtime_root().empty() ? native_client_root() : configured_runtime_root();
}
inline void configure_native_runtime(const std::filesystem::path& path) {
    if (!path.empty() && !path.is_absolute())
        throw std::runtime_error("C++ runtime path must be absolute");
    configured_runtime_root() = path;
}
inline std::function<void(const std::string&)>& prepare_embedded_backend() {
    static std::function<void(const std::string&)> prepare;
    return prepare;
}
inline void append_cpp_devices(std::vector<ExecutionDevice>& devices) {
    try {
        auto root = native_runtime_root();
        std::ifstream file(root / "phoenix_native_cpp.json");
        Json config;
        file >> config;
        auto validated = config.value("validated_devices", std::vector<std::string>{});
        for (const auto& device_config : config.value("devices", Json::array())) {
            auto device_id = device_config.at("id").get<std::string>();
            if (!valid_execution_device(device_id) || device_id == "native")
                continue;
            if (!device_dependencies_present(root, config, device_id))
                continue;
            bool tested =
                std::find(validated.begin(), validated.end(), device_id) != validated.end();
            if (!tested && !config.value("allow_unvalidated", false))
                continue;
            auto kind = device_id == "cpu" ? "cpu" : device_id.substr(0, device_id.find(':'));
            if (!std::filesystem::is_regular_file(root / ("phoenix_" + kind + "_cpp.dll")))
                continue;
            if (std::none_of(devices.begin(), devices.end(),
                             [&](const auto& entry) { return entry.id == device_id; }))
                devices.push_back({device_id, device_config.value("label", device_id) +
                                                  (tested ? " (C++)" : " (C++, untested)")});
        }
    } catch (const std::exception&) {
    }
}
class WorkerClient {
    struct State {
        phoenix_cpp::Handle cancellation{
            phoenix_cpp::require_handle(CreateEventW(nullptr, TRUE, FALSE, nullptr))};
        std::mutex mutex;
        std::condition_variable condition;
        bool stop = false, alive = true, initialized = false;
        WorkerState state = WorkerState::idle;
        DWORD pid = GetCurrentProcessId();
        uint64_t serial = 0, last = 0;
        uint32_t generation = 0;
        Json info, metadata;
        std::vector<float> color, motion, depth, validity, output, mask, ui, alpha, backbuffer, distortion;
    };
    std::shared_ptr<State> worker_state = std::make_shared<State>();
    size_t pixels;
    static std::atomic_bool& occupied() {
        static std::atomic_bool b = false;
        return b;
    }
    static void run(std::shared_ptr<State> worker_state, Json config, unsigned width,
                    unsigned height) {
        try {
            if (prepare_embedded_backend())
                prepare_embedded_backend()(config.at("backend").get<std::string>());
            auto root = std::filesystem::u8path(config.at("runtime_root").get<std::string>());
            phoenix_cpp::NativeSession session;
            session.set_cancellation(worker_state->cancellation);
            session.open(root, config, width, height);
            {
                std::lock_guard lock(worker_state->mutex);
                worker_state->initialized = true;
                worker_state->pid = session.process_id();
                worker_state->info = {
                    {"initialized", true},
                    {"runtime", "native-cpp"},
                    {"device", config["device"]},
                    {"inference_validated",
                     Json::parse(session.status_json()).value("inference_validated", false)}};
            }
            uint32_t last_generation = UINT32_MAX;
            uint64_t last_frame = 0;
            for (;;) {
                std::unique_lock lock(worker_state->mutex);
                worker_state->condition.wait(lock, [&] {
                    return worker_state->stop || worker_state->state == WorkerState::queued;
                });
                if (worker_state->stop)
                    break;
                auto metadata = worker_state->metadata;
                auto generation = worker_state->generation;
                auto serial = worker_state->serial;
                worker_state->state = WorkerState::running;
                lock.unlock();
                if (last_generation != generation ||
                    metadata.value("motion_from_frame", last_frame) != last_frame)
                    metadata["reset"] = true;
                std::vector<float> output(size_t(width) * height * 3);
                int result = session.frame_advanced(
                    worker_state->color.data(),
                    worker_state->motion.empty() ? nullptr : worker_state->motion.data(),
                    worker_state->depth.empty() ? nullptr : worker_state->depth.data(),
                    worker_state->validity.empty() ? nullptr : worker_state->validity.data(),
                    metadata.dump(), output.data(),
                    worker_state->mask.empty() ? nullptr : worker_state->mask.data(),
                    worker_state->ui.empty() ? nullptr : worker_state->ui.data(),
                    worker_state->alpha.empty() ? nullptr : worker_state->alpha.data(),
                    worker_state->backbuffer.empty() ? nullptr : worker_state->backbuffer.data(),
                    worker_state->distortion.empty() ? nullptr : worker_state->distortion.data());
                Json answer;
                if (result) {
                    answer = {{"ok", false}, {"error", session.error()}};
                } else {
                    answer = Json::parse(session.status_json());
                    answer["ok"] = true;
                    answer["finite"] =
                        std::all_of(output.begin(), output.end(),
                                    [](float sample_value) { return std::isfinite(sample_value); });
                    if (!answer["finite"].get<bool>()) {
                        answer["ok"] = false;
                        answer["error"] = "Nonfinite native network output";
                        result = -1;
                    }
                    answer["elapsed_ms"] = answer.value("worker_ms", 0.0);
                    answer["phase_ms"] = {
                        {"features_ms", answer.value("prepare_ms", 0.0)},
                        {"graph_roundtrip_ms", answer.value("execute_with_transfers_ms", 0.0)},
                        {"resolve_ms", answer.value("resolve_ms", 0.0)}};
                    answer["provider"] = config["backend"];
                    answer["kernel_engine"] = "native-cpp";
                    answer["frame_id"] = serial;
                    answer["generation"] = generation;
                    answer["budget_epoch"] = metadata.value("budget_epoch", uint64_t(0));
                    answer["requested_transformer_blocks"] = metadata.value("transformer_blocks", 8u);
                    answer["device"] = config["device"];
                    answer["requested_device"] = config["device"];
                }
                lock.lock();
                worker_state->info = std::move(answer);
                worker_state->output = std::move(output);
                worker_state->state = result ? WorkerState::error : WorkerState::result;
                last_generation = generation;
                last_frame = serial;
            }
        } catch (const std::exception& exception) {
            std::lock_guard lock(worker_state->mutex);
            worker_state->info = {{"ok", false}, {"error", exception.what()}};
            worker_state->state = WorkerState::error;
            worker_state->initialized = true;
        }
        {
            std::lock_guard lock(worker_state->mutex);
            worker_state->alive = false;
        }
        occupied() = false;
    }

  public:
    WorkerClient(unsigned width, unsigned height, const std::filesystem::path&,
                 const std::filesystem::path&, const std::filesystem::path&,
                 const std::wstring& device, const std::filesystem::path&,
                 const std::filesystem::path& = {}, const std::wstring& = L"auto",
                 const std::filesystem::path& = {}, const std::filesystem::path& = {},
                 const phoenix_model::WeightSource& source = {}, unsigned cpu_threads = 0)
        : pixels(size_t(width) * height) {
        if (!width || !height || width > 8192 || height > 8192)
            throw std::runtime_error("Native frame size unsupported");
        auto root = native_runtime_root();
        std::ifstream f(root / "phoenix_native_cpp.json");
        if (!f)
            throw std::runtime_error("Missing phoenix_native_cpp.json in the C++ runtime folder");
        Json config;
        f >> config;
        config["runtime_root"] = phoenix_cpp::path_utf8(root);
        std::string device_id(device.begin(), device.end());
        if (!valid_execution_device(device_id) || device_id == "native")
            throw std::runtime_error("Invalid processing device");
        auto kind = device_id == "cpu" ? "cpu" : device_id.substr(0, device_id.find(':'));
        config["backend"] = kind;
        config["device"] = device_id;
        if (source.external() || source.model_id != phoenix_model::models::original.id)
            config["weight_source"] = source.configuration();
        config["cpu_threads"] = cpu_threads;
        if (occupied().exchange(true))
            throw std::runtime_error("Previous native execution is still retiring");
        try {
            std::thread([state = worker_state, config, width, height] {
                run(state, config, width, height);
            }).detach();
        } catch (...) {
            occupied() = false;
            throw;
        }
    }
    ~WorkerClient() {
        std::lock_guard lock(worker_state->mutex);
        worker_state->stop = true;
        SetEvent(worker_state->cancellation);
        worker_state->condition.notify_all();
    }
    bool alive() const {
        std::lock_guard lock(worker_state->mutex);
        return worker_state->alive;
    }
    DWORD process_id() const {
        std::lock_guard lock(worker_state->mutex);
        return worker_state->pid;
    }
    DWORD exit_code() const {
        return alive() ? STILL_ACTIVE : 0;
    }
    Json status() {
        std::lock_guard lock(worker_state->mutex);
        auto result = worker_state->info;
        result["state"] = uint32_t(worker_state->state);
        result["initialized"] = worker_state->initialized;
        return result;
    }
    bool submit(uint64_t serial, uint32_t generation, const float* color, const float* motion,
                const float* depth, const Json& metadata, const float* valid = nullptr,
                const float* mask = nullptr, const float* ui = nullptr, const float* alpha = nullptr,
                const float* backbuffer = nullptr, const float* distortion = nullptr) {
        std::unique_lock lock(worker_state->mutex, std::try_to_lock);
        if (!lock || !worker_state->alive || !worker_state->initialized ||
            worker_state->state != WorkerState::idle)
            return false;
        if (!color || serial <= worker_state->last)
            throw std::runtime_error("Invalid native frame sequence");
        worker_state->color.assign(color, color + pixels * 3);
        worker_state->motion.clear();
        worker_state->depth.clear();
        worker_state->validity.clear();
        if (motion)
            worker_state->motion.assign(motion, motion + pixels * 2);
        if (depth)
            worker_state->depth.assign(depth, depth + pixels);
        if (valid)
            worker_state->validity.assign(valid, valid + pixels);
        auto snapshot = [&](std::vector<float>& target, const float* source, unsigned channels) {
            if (source) target.assign(source, source + pixels * channels); else target.clear();
        };
        snapshot(worker_state->mask,mask,4); snapshot(worker_state->ui,ui,4);
        snapshot(worker_state->alpha,alpha,1); snapshot(worker_state->backbuffer,backbuffer,3);
        snapshot(worker_state->distortion,distortion,4);
        worker_state->metadata = metadata;
        worker_state->serial = worker_state->last = serial;
        worker_state->generation = generation;
        worker_state->state = WorkerState::queued;
        worker_state->condition.notify_one();
        return true;
    }
    bool poll(uint64_t& serial, uint32_t& generation, std::vector<float>& output, Json& metadata) {
        std::unique_lock lock(worker_state->mutex, std::try_to_lock);
        if (!lock || (worker_state->state != WorkerState::result &&
                      worker_state->state != WorkerState::error))
            return false;
        serial = worker_state->serial;
        generation = worker_state->generation;
        metadata = worker_state->info;
        if (worker_state->state == WorkerState::result)
            output = std::move(worker_state->output);
        else
            output.clear();
        worker_state->state = WorkerState::idle;
        return true;
    }
};
}
