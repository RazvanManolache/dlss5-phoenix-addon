// An asynchronous, architecture-neutral connection to our own 64-bit helper.
// Pipe I/O, DLL loading and inference never run on the game's render thread.
#pragma once
#include "../../native/support/embedded_runtime.h"
#include "../../native/support/model_tools.h"
#include "../../native/support/routing.h"
#include "session_policy.h"
#include "../reshade/panel_contract.h"
#include "protocol.h"
#include "owned_worker.h"
#include "portable_policy.h"
#include "processing_pixels.h"
#include "wow64/bridge_policy.h"
#include "wow64/graph32.h"
#include "../../../native_runtime/runtime_api.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <d3d11_4.h>
#include <deque>
#include <mutex>
#include <source_location>
#include <thread>
#include <vector>
#include <windows.h>
#include <wrl/client.h>

namespace phoenix_compat {
using Json = nlohmann::json;
namespace fs = std::filesystem;
template <class T> using Ptr = Microsoft::WRL::ComPtr<T>;
inline void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
inline void hr(HRESULT result, std::source_location location = std::source_location::current()) {
    if (FAILED(result))
        throw std::runtime_error("Graphics bridge HRESULT " + std::to_string(uint32_t(result)) +
                                 " at " + location.function_name() + ":" +
                                 std::to_string(location.line()));
}
struct Handle {
    HANDLE value = nullptr;
    ~Handle() {
        if (value && value != INVALID_HANDLE_VALUE)
            CloseHandle(value);
    }
    Handle() = default;
    Handle(const Handle&) = delete;
};
inline FeedBuildAck duplicate_fence_ack(HANDLE process, const FeedBuildAck& acknowledgement) {
    // The helper retains its own handle ownership. Create game-local
    // duplicates only here, so cancelled/CPU handshakes cannot orphan
    // handles in the game process.
    Handle input, output;
    require(DuplicateHandle(process, HANDLE(uintptr_t(acknowledgement.fence_in)),
                            GetCurrentProcess(), &input.value, 0, FALSE,
                            DUPLICATE_SAME_ACCESS) != 0,
            "Cannot duplicate shared input fence");
    require(DuplicateHandle(process, HANDLE(uintptr_t(acknowledgement.fence_out)),
                            GetCurrentProcess(), &output.value, 0, FALSE,
                            DUPLICATE_SAME_ACCESS) != 0,
            "Cannot duplicate shared output fence");
    auto result = acknowledgement;
    result.fence_in = uint64_t(uintptr_t(input.value));
    result.fence_out = uint64_t(uintptr_t(output.value));
    input.value = output.value = nullptr; // Caller owns the returned handles.
    return result;
}
inline uint64_t acknowledged_clear_revision(uint64_t revision, FeedUiAction action,
                                            uint32_t accepted) {
    return revision +
           (accepted == 1 && action.action == uint32_t(phoenix_ui::Action::ClearTiming) ? 1 : 0);
}
// Delete only our nonce-named child directory, after helper termination. Never
// follow redirects into the cache/game, and preserve capture directories/files.
inline void cleanup_stopped_session(const fs::path& folder, const fs::path& expected_parent,
                                    const std::string& expected_name) noexcept try {
    std::error_code error;
    if (folder.empty() || folder.filename() != fs::u8path(expected_name) ||
        fs::weakly_canonical(folder, error) != expected_parent / fs::u8path(expected_name) || error)
        return;
    for (const auto& item : fs::directory_iterator(folder, error)) {
        if (error || !item.is_regular_file(error) || error || item.is_symlink(error))
            return;
        const auto extension = item.path().extension().string();
        if (extension != ".json" && extension != ".ini" && extension != ".log" &&
            extension != ".dll" && extension != ".addon64" && extension != ".exe")
            return;
    }
    if (!error)
        fs::remove_all(folder, error);
} catch (...) {
    // Session deletion is best effort. Filesystem races and access failures
    // must never escape the detached worker and terminate the game process.
}
enum class Phase : unsigned { Starting, OpenFences, Ready, Busy, Finished, Failed, Stopped };
class HostState : public std::enable_shared_from_this<HostState> {
    std::mutex mutex_;
    std::condition_variable changed_;
    std::atomic<bool> stopping_{false};
    std::atomic<Phase> phase_{Phase::Starting};
    Handle job_, process_, pipe_, fence_event_, io_event_;
    Ptr<ID3D11Fence> output_fence_;
    FeedBuildAck acknowledgement_{};
    std::atomic<DWORD> child_pid_{0};
    bool portable_ = false;
    bool native_ = false;
    std::string native_mode_;
    FeedFrameMsg request_{};
    std::vector<uint8_t> input_, output_;
    std::string error_;
    double roundtrip_ms_ = 0;
    uint64_t completed_at_ = 0;
    Json ui_snapshot_ = Json::object();
    std::deque<FeedUiAction> ui_actions_;
    bool model_enabled_ = true;
    Settings tuning_settings_;
    std::optional<std::string> pending_tuning_;
    uint64_t tuning_revision_ = 0, applied_tuning_revision_ = 0,
             completed_tuning_revision_ = 0, taken_tuning_revision_ = 0;
    uint64_t session_id_ = 0, ui_clear_revision_ = 0;
    bool capture_requested_ = false;
    fs::path session_parent_;
    std::string session_name_;

    void serve_portable(phoenix_model::WeightSource source, Settings settings,
                        FeedBuild build, PortableOptions options) {
        try {
            // Providers retain process-global device state. A cancelled worker
            // must finish before a new canvas is allowed to initialize it.
            static std::mutex provider_owner;
            std::unique_lock provider_lock(provider_owner, std::defer_lock);
            const auto retirement_deadline = GetTickCount64() + 30000;
            while (!provider_lock.try_lock()) {
                require(!stopping_, "Portable execution cancelled while waiting for the previous session");
                require(GetTickCount64() < retirement_deadline, "Previous portable session did not retire within 30 seconds");
                Sleep(5);
            }
            require(!stopping_, "Portable execution cancelled");
            require(build.cpu_bridge && build.width == build.output_width &&
                        build.height == build.output_height,
                    "Portable execution requires same-size pixel transport");
            require(!needs_nvidia_dll(settings), "NVIDIA DLSS5 evaluation requires the helper");
            require(settings.resolution_mode == 0 && !settings.host_unjitter,
                    "Portable backbuffer execution requires normal resolution and no unjitter");
            require(options.root.is_absolute(), "Portable runtime path must be absolute");
            source = phoenix_model_tools::selection(settings, source);
            require(source.external(), "No model selected for portable execution");
            Json config = Json::object();
            std::ifstream file(options.root / "phoenix_native_cpp.json");
            if (file) file >> config;
            require(config.is_object(), "Invalid portable runtime configuration");
            Json devices = Json::array();
            // Optional SDKs and device enumeration stay off the presentation thread.
            if (options.device == "auto") {
                const auto provider = options.root / "vulkan" / "phoenix_vulkan.dll";
                struct Library {
                    HMODULE value;
                    ~Library() { if (value) FreeLibrary(value); }
                } library{LoadLibraryExW(provider.c_str(), nullptr,
                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32)};
                require(library.value != nullptr, "Matching-bitness Vulkan provider unavailable");
                using Count = int(__cdecl*)();
                using Info = int(__cdecl*)(unsigned, char*, char*, unsigned*, unsigned*);
                const auto count = reinterpret_cast<Count>(GetProcAddress(library.value, "phoenix_device_count"));
                const auto info = reinterpret_cast<Info>(GetProcAddress(library.value, "phoenix_device_info"));
                require(count && info, "Portable Vulkan discovery ABI unavailable");
                const int total = count();
                require(total >= 0 && total <= 64, "Invalid portable device count");
                for (int index = 0; index < total; ++index) {
                    char uuid[33]{}, name[256]{};
                    unsigned vendor = 0, capable = 0;
                    if (info(unsigned(index), uuid, name, &vendor, &capable) || !capable) continue;
                    uuid[32] = 0; name[255] = 0;
                    auto id = "vulkan:" + std::string(uuid);
                    if (phoenix_addon::phoenix::valid_execution_device(id))
                        devices.push_back({{"id", id}, {"label", std::string(name) + " (portable Vulkan)"}});
                }
                require(!devices.empty(), "No capable portable Vulkan device found");
                options.device = devices.front().at("id").get<std::string>();
            }
            config["runtime_root"] = phoenix_model::source_path_string(options.root);
            config["backend"] = portable_backend(options.device);
            config["device"] = options.device;
            config["weight_source"] = source.configuration();
            config["cpu_threads"] = settings.cpu_threads;
            if (options.allow_unvalidated) config["allow_unvalidated"] = true;
            phoenix_cpp::RuntimeApi runtime;
            const auto working = settings.processing.enabled
                ? phoenix_addon::phoenix::processing_resolution(build.width,build.height,settings.processing,false)
                : phoenix_addon::phoenix::fit_resolution(build.width,build.height,build.width,build.height);
            runtime.open(options.root, config.dump(), working.image_width, working.image_height);
            auto snapshot = [&](Json status) {
                const auto schedule = status.value("network_schedule", Json::object());
                return Json{{"schema", "phoenix-panel/1"}, {"runtime", status},
                    {"transformer_blocks", schedule.value("transformer_blocks", 8u)},
                    {"budget_estimate_ms", schedule.value("estimate_ms", 0.0)},
                    {"runtime_root", phoenix_model::source_path_string(options.root)},
                    {"processing_location", "in_process_portable"},
                    {"inference_pid", GetCurrentProcessId()}, {"device", options.device},
                    {"execution_devices", devices}, {"ui_clear_revision", ui_clear_revision_},
                    {"nvidia", {{"ready", false}, {"status", "NVIDIA DLSS5 modes use the retained helper"}}}};
            };
            std::unique_lock lock(mutex_);
            ui_snapshot_ = snapshot(Json::parse(runtime.status_json()));
            phase_ = Phase::Ready;
            while (!stopping_) {
                changed_.wait(lock, [&] { return stopping_ || phase_ == Phase::Busy ||
                    pending_tuning_ || !ui_actions_.empty(); });
                if (stopping_) break;
                if (pending_tuning_) {
                    settings = tuning_settings_;
                    pending_tuning_.reset();
                    applied_tuning_revision_ = tuning_revision_;
                }
                for (const auto action : ui_actions_) {
                    if (action.action == uint32_t(phoenix_ui::Action::ClearTiming)) {
                        ++ui_clear_revision_;
                        ui_snapshot_["ui_clear_revision"] = ui_clear_revision_;
                        ui_snapshot_["frame_wall_ms"] = 0;
                        roundtrip_ms_ = 0;
                    } else if (action.action == uint32_t(phoenix_ui::Action::Marker))
                        ui_snapshot_["markers"] = ui_snapshot_.value("markers", uint64_t(0)) + 1;
                    // Backbuffer frames always reset history: no invented motion guides.
                }
                ui_actions_.clear();
                if (phase_ != Phase::Busy) continue;
                const auto request = request_;
                const auto revision = applied_tuning_revision_;
                const auto frame_settings = settings;
                auto pixels = std::move(input_);
                lock.unlock();
                require(pixels.size() == size_t(build.width) * build.height * 4,
                        "Portable input extent mismatch");
                const auto started = std::chrono::steady_clock::now();
                auto color = portable_color(pixels);
                std::optional<pixels::Frame> processing;
                if (frame_settings.processing.enabled) {
                    processing.emplace(color,build.width,build.height,frame_settings.processing);
                    color=processing->encoded;
                }
                std::vector<float> enhanced(color.size());
                auto metadata = portable_metadata(frame_settings, true);
                if (metadata.contains("adaptive_budget"))
                    require(Json::parse(runtime.status_json()).value("capabilities", Json::object()).contains("granular_controls"),
                            "Adaptive portable controls require the matching updated runtime");
                for (unsigned iteration = 0; iteration < request.iterations; ++iteration) {
                    require(!stopping_, "Portable execution cancelled");
                    if (iteration) color = enhanced;
                    if (runtime.frame(runtime.instance, color.data(), nullptr, nullptr,
                                nullptr, metadata.dump().c_str(), enhanced.data()) != 0)
                        throw std::runtime_error(runtime.error());
                }
                if (processing) enhanced=processing->finish(std::move(enhanced));
                auto result = portable_compose(pixels, enhanced, build.width, frame_settings);
                auto status = snapshot(Json::parse(runtime.status_json()));
                const auto wall_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
                status["frame_wall_ms"] = wall_ms;
                lock.lock();
                output_ = std::move(result);
                ui_snapshot_ = std::move(status);
                roundtrip_ms_ = wall_ms;
                completed_at_ = GetTickCount64();
                completed_tuning_revision_ = revision;
                phase_ = Phase::Finished;
            }
            phase_ = Phase::Stopped;
        } catch (const std::exception& error) {
            std::lock_guard lock(mutex_);
            error_ = std::string("In-process portable: ") + error.what();
            phase_ = stopping_ ? Phase::Stopped : Phase::Failed;
        }
    }

    void serve_native(phoenix_model::WeightSource source,Settings settings,FeedBuild build,NativeOptions options) {
        try {
#ifdef _WIN64
            throw std::runtime_error("WOW64 execution is available only in the x86 addon");
#else
            // Process-local native export/compatibility state must have one owner.
            // Retiring workers retain their packets and keep the addon pinned.
            static std::mutex native_owner;
            std::unique_lock owner(native_owner,std::defer_lock);
            const auto deadline=GetTickCount64()+30000;
            while(!owner.try_lock()) {require(!stopping_&&GetTickCount64()<deadline,"Previous native session is still retiring");Sleep(5);}
            require(!stopping_,"Native execution cancelled");
            require(build.cpu_bridge&&build.width==build.output_width&&build.height==build.output_height,"Native execution requires same-size pixel transport");
            validate_native_settings(settings,options.driver_host);
            source=phoenix_model_tools::selection(settings, source);
            require(source.external(),"No model selected for native execution");
            const auto bundle=options.root/"native";
            const auto working = settings.processing.enabled
                ? phoenix_addon::phoenix::processing_resolution(build.width,build.height,settings.processing,false)
                : phoenix_addon::phoenix::fit_resolution(build.width,build.height,build.width,build.height);
            phoenix_wow64::FrameSession renderer(options.root,bundle,source,working.image_width,working.image_height,settings,options.driver_host);
            std::unique_lock lock(mutex_);
            auto snapshot=[&](double wall) {return Json{{"schema","phoenix-panel/1"},{"processing_location",native_mode_},
                {"inference_pid",GetCurrentProcessId()},{"helper_pid",0},{"frame_wall_ms",wall},{"ui_clear_revision",ui_clear_revision_},
                {"nvidia",renderer.nvidia_status()},
                {"runtime",{{"backend",native_mode_},{"gpu_processing_ms",renderer.gpu_ms()}}}};};
            ui_snapshot_=snapshot(0);phase_=Phase::Ready;
            while(!stopping_) {
                changed_.wait(lock,[&]{return stopping_||phase_==Phase::Busy||pending_tuning_||!ui_actions_.empty();});
                if(stopping_) break;
                if(pending_tuning_) {settings=tuning_settings_;pending_tuning_.reset();applied_tuning_revision_=tuning_revision_;}
                for(const auto& action:ui_actions_) {
                    if(action.action==uint32_t(phoenix_ui::Action::ClearTiming)) {++ui_clear_revision_;roundtrip_ms_=0;}
                }
                ui_actions_.clear();
                if(phase_!=Phase::Busy) continue;
                const auto request=request_; const auto revision=applied_tuning_revision_; const auto frame_settings=settings;
                auto pixels=std::move(input_);lock.unlock();
                const auto started=std::chrono::steady_clock::now();
                auto phi=portable_color(pixels);std::vector<float> nv;
                std::optional<pixels::Frame> processing;
                std::vector<uint8_t> working_pixels;
                if(frame_settings.processing.enabled) {
                    processing.emplace(phi,build.width,build.height,frame_settings.processing);
                    working_pixels.resize(size_t(processing->working_width)*processing->working_height*4,255);
                }
                auto evaluate=[&](unsigned backend) {
                    std::vector<float> result=processing?processing->encoded:std::vector<float>{};
                    const unsigned passes=request.iterations*unsigned(frame_settings.refinement_passes);
                    for(unsigned i=0;i<passes;++i) {
                        require(!stopping_,"Native execution cancelled");
                        result=renderer.evaluate(processing?working_pixels:pixels,frame_settings,backend,processing||i?&result:nullptr);
                    }
                    return processing?processing->finish(std::move(result)):result;
                };
                if(phoenix_addon::phoenix::uses_ours(frame_settings.mode)) phi=evaluate(3);
                if(needs_nvidia_dll(frame_settings)) nv=evaluate(1);
                auto result=portable_compose(pixels,phi,build.width,frame_settings,nv.empty()?nullptr:&nv);
                const auto wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
                auto status=snapshot(wall);lock.lock();output_=std::move(result);ui_snapshot_=std::move(status);
                roundtrip_ms_=wall;completed_at_=GetTickCount64();completed_tuning_revision_=revision;phase_=Phase::Finished;
            }
            lock.unlock();
            renderer.retire();
            phase_=Phase::Stopped;
#endif
        } catch(const std::exception& error) {std::lock_guard lock(mutex_);error_=std::string("Native x86 processing: ")+error.what();phase_=stopping_?Phase::Stopped:Phase::Failed;}
    }

    void transfer(void* data, size_t bytes, bool write) {
        auto* cursor = static_cast<uint8_t*>(data);
        while (bytes) {
            require(!stopping_, "Graphics helper cancelled");
            DWORD transferred = 0;
            const auto count = DWORD((std::min)(bytes, size_t(1) << 20));
            OVERLAPPED operation{};
            operation.hEvent = io_event_.value;
            ResetEvent(operation.hEvent);
            auto ok = write ? WriteFile(pipe_.value, cursor, count, &transferred, &operation)
                            : ReadFile(pipe_.value, cursor, count, &transferred, &operation);
            if (ok)
                ok = GetOverlappedResult(pipe_.value, &operation, &transferred, FALSE);
            else if (GetLastError() == ERROR_IO_PENDING) {
                const HANDLE waits[]{io_event_.value, process_.value};
                const auto deadline = GetTickCount64() + 60000;
                DWORD waited = WAIT_TIMEOUT;
                while (!stopping_ && GetTickCount64() < deadline && waited == WAIT_TIMEOUT)
                    waited = WaitForMultipleObjects(2, waits, FALSE, 50);
                if (waited != WAIT_OBJECT_0) {
                    CancelIoEx(pipe_.value, &operation);
                    // Retire the cancelled I/O before its OVERLAPPED leaves scope.
                    GetOverlappedResult(pipe_.value, &operation, &transferred, TRUE);
                    throw std::runtime_error("Graphics helper pipe cancelled, exited or timed out");
                }
                ok = GetOverlappedResult(pipe_.value, &operation, &transferred, FALSE);
            }
            require(ok && transferred, "Graphics helper pipe disconnected");
            cursor += transferred;
            bytes -= transferred;
        }
    }
    void prepare(HMODULE addon, const phoenix_model::WeightSource& source,
                 phoenix_addon::phoenix::Settings settings, fs::path& folder) {
        auto cache = phoenix_embedded::install(addon, {}, "compatibility", &stopping_);
        require(!cache.root.empty(), "This addon build has no embedded graphics helper");
        session_parent_ = fs::weakly_canonical(cache.root.parent_path().parent_path()) / "sessions";
        fs::create_directories(session_parent_);
        session_parent_ = fs::canonical(session_parent_);
        session_name_ = std::to_string(GetCurrentProcessId()) + "-" + std::to_string(session_id_);
        folder = session_parent_ / fs::u8path(session_name_);
        fs::create_directories(folder);
        for (const wchar_t* name :
             {L"phoenix_graphics_host.exe", L"dxgi.dll", L"Phoenix.addon64"}) {
            require(!stopping_, "Graphics helper cancelled");
            auto original = cache.root / "compatibility" / name;
#ifdef _WIN64
            if (std::wstring(name) == L"Phoenix.addon64" && !fs::exists(original)) {
                wchar_t module[32768]{};
                require(GetModuleFileNameW(addon, module, 32768) != 0, "Addon path unavailable");
                // Cache this x64 addon once per version. Linking a game on E:
                // straight into a session on C: would fall back to copying the
                // entire addon on every start. Sessions can hard-link this copy.
                original = cache.root / "self" / name;
                const auto digest = phoenix_embedded::file_digest(module, &stopping_);
                require(!digest.empty(), "Cannot verify the running addon");
                if (phoenix_embedded::file_digest(original, &stopping_) != digest) {
                    fs::create_directories(original.parent_path());
                    auto temporary = original;
                    temporary += L".tmp-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
                                 std::to_wstring(session_id_);
                    fs::copy_file(module, temporary);
                    require(MoveFileExW(temporary.c_str(), original.c_str(),
                                        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0,
                            "Cannot publish the cached x64 addon");
                }
            }
#endif
            require(fs::is_regular_file(original), "Embedded 64-bit helper component is missing");
            const auto destination = folder / name;
            if (!CreateHardLinkW(destination.c_str(), original.c_str(), nullptr))
                fs::copy_file(original, destination, fs::copy_options::overwrite_existing);
        }
        Json config = {{"schema", "phoenix-live/1"},
                       {"armed", true},
                       {"input_provider", "feeder_d3d12"},
                       {"sandbox", phoenix_model::source_path_string(folder)},
                       {"in_place_game_directory", phoenix_model::source_path_string(folder)},
                       {"in_place_approval", "I approve driving this installed game in place"},
                       {"direct_transfer", "srgb"},
                       {"color_domain", "game_nr_sdr_proxy"},
                       {"frame_limit", 4000000},
                       {"weight_source", source.configuration()}};
        std::ofstream config_file(folder / "Phoenix.json");
        config_file << config.dump(2);
        config_file.close();
        require(bool(config_file), "Cannot write helper configuration");
        // The companion has a full Phoenix panel, but startup settings come from
        // the game's panel. Restarting applies model/device/geometry changes.
        std::ofstream ini(folder / "ReShade.ini");
        ini << "[GENERAL]\nEffectSearchPaths=\nPresetPath=ReShadePreset.ini\n\n[Phoenix]\n";
        auto helper_settings = settings;
        if (!helper_settings.nvidia_dll_folder.empty()) {
            auto folder_path = fs::u8path(helper_settings.nvidia_dll_folder);
            if (!folder_path.is_absolute()) folder_path = phoenix_model_tools::local_library_directory() / folder_path;
            helper_settings.nvidia_dll_folder = phoenix_model::source_path_string(folder_path.lexically_normal());
        }
        phoenix_addon::phoenix::visit_settings(helper_settings, [&](const char* name, const auto& value) {
            Json encoded = value;
            ini << name << '='
                << (encoded.is_string()    ? encoded.template get<std::string>()
                    : encoded.is_boolean() ? (encoded.template get<bool>() ? "1" : "0")
                                           : encoded.dump())
                << '\n';
        });
        require(bool(ini), "Cannot write helper settings");
    }
    void serve(HMODULE addon, phoenix_model::WeightSource source,
               phoenix_addon::phoenix::Settings settings, FeedBuild build) {
        fs::path folder;
        const char* stage = "model selection";
        try {
            source = phoenix_model_tools::selection(settings, source);
            require(source.external(),
                    "No model selected. Browse to your NVIDIA DLSS5 DLL or a compatible model folder.");
            phoenix_addon::phoenix::normalize_direct_comparison(settings);
            stage = "embedded helper installation";
            prepare(addon, source, settings, folder);
            stage = "helper process startup";
            require(!stopping_, "Graphics helper cancelled");
            job_.value = CreateJobObjectW(nullptr, nullptr);
            require(job_.value != nullptr, "Cannot create helper job");
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            require(SetInformationJobObject(job_.value, JobObjectExtendedLimitInformation, &limits,
                                            sizeof(limits)) != 0,
                    "Cannot contain helper");
            const auto executable = folder / L"phoenix_graphics_host.exe";
            std::wstring command = L"\"" + executable.wstring() + L"\" " +
                                   std::to_wstring(GetCurrentProcessId()) + L" --hide --adapter " +
                                   std::to_wstring(build.adapter_luid) + L" --session " +
                                   std::to_wstring(session_id_);
            STARTUPINFOW startup{};
            startup.cb = sizeof(startup);
            startup.dwFlags = STARTF_USESHOWWINDOW;
            startup.wShowWindow = SW_HIDE;
            PROCESS_INFORMATION child{};
            require(CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, FALSE,
                                   CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, folder.c_str(),
                                   &startup, &child) != 0,
                    "Cannot start 64-bit graphics helper");
            process_.value = child.hProcess;
            child_pid_.store(child.dwProcessId);
            struct StopChild {
                HANDLE job, process;
                ~StopChild() {
                    TerminateJobObject(job, 0);
                    WaitForSingleObject(process, 5000);
                }
            } cleanup{job_.value, process_.value};
            if (!AssignProcessToJobObject(job_.value, child.hProcess)) {
                TerminateProcess(child.hProcess, 1);
                CloseHandle(child.hThread);
                throw std::runtime_error("Cannot contain graphics helper process");
            }
            ResumeThread(child.hThread);
            CloseHandle(child.hThread);
            char pipe_name[128]{};
            sprintf_s(pipe_name, FEED_SESSION_PIPE_FMT, GetCurrentProcessId(),
                      static_cast<unsigned long long>(session_id_));
            const auto deadline = GetTickCount64() + 30000;
            while (!stopping_ && GetTickCount64() < deadline) {
                pipe_.value = CreateFileA(pipe_name, GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                                          OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
                if (pipe_.value != INVALID_HANDLE_VALUE)
                    break;
                pipe_.value = nullptr;
                require(WaitForSingleObject(process_.value, 20) == WAIT_TIMEOUT,
                        "Graphics helper exited at startup");
            }
            require(pipe_.value != nullptr, "Graphics helper connection timed out");
            ULONG server = 0;
            require(GetNamedPipeServerProcessId(pipe_.value, &server) &&
                        server == child.dwProcessId,
                    "Graphics pipe belongs to a different process");
            FeedHello hello{FEED_IPC_MAGIC, FEED_IPC_VERSION, GetCurrentProcessId()};
            FeedHelloAck greeting{};
            transfer(&hello, sizeof(hello), true);
            transfer(&greeting, sizeof(greeting), false);
            require(greeting.magic == FEED_IPC_MAGIC && greeting.version == FEED_IPC_VERSION,
                    "Graphics helper protocol mismatch");
            uint8_t tag = 'B';
            stage = "frame resource handshake";
            transfer(&tag, 1, true);
            transfer(&build, sizeof(build), true);
            transfer(&acknowledgement_, sizeof(acknowledgement_), false);
            require(acknowledgement_.ok == 1,
                    "64-bit helper rejected frame resources; see its log");
            phase_ = build.cpu_bridge ? Phase::Ready : Phase::OpenFences;
            stage = "frame processing";
            std::unique_lock lock(mutex_);
            uint64_t ui_tick = 0;
            while (!stopping_) {
                changed_.wait_for(lock, std::chrono::milliseconds(100), [&] {
                    return stopping_ || phase_ == Phase::Busy || !ui_actions_.empty() || pending_tuning_;
                });
                if (stopping_)
                    break;
                if (pending_tuning_) {
                    auto patch = std::move(*pending_tuning_);
                    pending_tuning_.reset();
                    const auto revision = tuning_revision_;
                    lock.unlock();
                    tag = 'S';
                    FeedSettingsPatch message{uint32_t(patch.size())};
                    transfer(&tag, 1, true);
                    transfer(&message, sizeof(message), true);
                    transfer(patch.data(), patch.size(), true);
                    uint32_t accepted = 0;
                    transfer(&accepted, sizeof(accepted), false);
                    require(accepted == 1, "Graphics helper rejected live controls");
                    lock.lock();
                    applied_tuning_revision_ = revision;
                    continue; // Coalesce edits received during the transfer before the next frame.
                }
                // Query at most once per second. Never serialize histories or
                // wait on pipe I/O inside the game/UI thread.
                if (!ui_actions_.empty() || GetTickCount64() - ui_tick >= 1000) {
                    auto actions = std::move(ui_actions_);
                    ui_actions_.clear();
                    lock.unlock();
                    bool action_rejected = false;
                    for (auto action : actions) {
                        tag = 'A';
                        transfer(&tag, 1, true);
                        transfer(&action, sizeof(action), true);
                        uint32_t accepted = 0;
                        transfer(&accepted, sizeof(accepted), false);
                        action_rejected |= accepted != 1;
                        ui_clear_revision_ =
                            acknowledged_clear_revision(ui_clear_revision_, action, accepted);
                        capture_requested_ |=
                            accepted == 1 &&
                            action.action == uint32_t(phoenix_ui::Action::CaptureInputs);
                    }
                    tag = 'U';
                    transfer(&tag, 1, true);
                    FeedUiReply reply{};
                    transfer(&reply, sizeof(reply), false);
                    require(reply.bytes && reply.bytes <= 1024u * 1024u,
                            "Invalid helper UI snapshot size");
                    std::string text(reply.bytes, '\0');
                    transfer(text.data(), text.size(), false);
                    auto snapshot = Json::parse(text);
                    require(snapshot.is_object(), "Invalid helper UI snapshot");
                    snapshot["ui_clear_revision"] = ui_clear_revision_;
                    if (action_rejected)
                        snapshot["ui_action_error"] = "Helper rejected the UI action; see its "
                                                      "status. Inference remains active.";
                    lock.lock();
                    ui_snapshot_ = std::move(snapshot);
                    ui_tick = GetTickCount64();
                }
                if (phase_ != Phase::Busy) {
                    require(WaitForSingleObject(process_.value, 0) == WAIT_TIMEOUT,
                            "Graphics helper exited");
                    continue;
                }
                auto request = request_;
                const auto frame_tuning_revision = applied_tuning_revision_;
                auto pixels = std::move(input_);
                auto fence = output_fence_;
                lock.unlock();
                auto started = std::chrono::steady_clock::now();
                tag = 'F';
                transfer(&tag, 1, true);
                transfer(&request, sizeof(request), true);
                std::vector<uint8_t> result;
                if (build.cpu_bridge) {
                    transfer(pixels.data(), pixels.size(), true);
                    FeedCpuFrameAck reply{};
                    transfer(&reply, sizeof(reply), false);
                    require(reply.ok == 1 && reply.row_bytes == build.output_width * 4,
                            "Graphics helper rejected CPU frame");
                    result.resize(size_t(reply.row_bytes) * build.output_height);
                    transfer(result.data(), result.size(), false);
                } else {
                    require(fence != nullptr, "Shared output fence not opened");
                    hr(fence->SetEventOnCompletion(request.n, fence_event_.value));
                    const HANDLE waits[] = {fence_event_.value, process_.value};
                    DWORD waited = WAIT_TIMEOUT;
                    const auto until = GetTickCount64() + 60000;
                    while (!stopping_ && GetTickCount64() < until && waited == WAIT_TIMEOUT)
                        waited = WaitForMultipleObjects(2, waits, FALSE, 50);
                    require(waited == WAIT_OBJECT_0 && fence->GetCompletedValue() >= request.n &&
                                fence->GetCompletedValue() != UINT64_MAX,
                            "Shared output did not retire");
                }
                lock.lock();
                output_ = std::move(result);
                roundtrip_ms_ = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - started)
                                    .count();
                completed_at_ = GetTickCount64();
                completed_tuning_revision_ = frame_tuning_revision;
                phase_ = Phase::Finished;
            }
        } catch (const std::exception& error) {
            std::lock_guard lock(mutex_);
            error_ = std::string(stage) + ": " + error.what();
            if (!folder.empty())
                error_ += "; helper log: " +
                          phoenix_model::source_path_string(folder / "dlss5-feed-host.log");
            phase_ = stopping_ ? Phase::Stopped : Phase::Failed;
        }
        if (stopping_ && !capture_requested_ && process_.value &&
            WaitForSingleObject(process_.value, 0) == WAIT_OBJECT_0) {
            if (pipe_.value && pipe_.value != INVALID_HANDLE_VALUE) {
                CloseHandle(pipe_.value);
                pipe_.value = nullptr;
            }
            cleanup_stopped_session(folder, session_parent_, session_name_);
        }
    }

  public:
    HostState(const phoenix_addon::phoenix::Settings& settings) {
        model_enabled_ = settings.uplift;
        tuning_settings_ = settings;
        require(BCryptGenRandom(nullptr, reinterpret_cast<PUCHAR>(&session_id_),
                                sizeof(session_id_), BCRYPT_USE_SYSTEM_PREFERRED_RNG) >= 0 &&
                    session_id_ != 0,
                "Cannot create unique graphics session identity");
        // The worker can outlive a render-facing client. Pin the code that owns
        // its state/destructors, as previous synchronous clients already did.
        HMODULE pin = nullptr;
        require(GetModuleHandleExW(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                    reinterpret_cast<LPCWSTR>(&phoenix_embedded::default_cache), &pin) != 0,
                "Cannot retain graphics bridge module");
        fence_event_.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        io_event_.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        require(io_event_.value != nullptr, "Cannot create helper I/O event");
        require(fence_event_.value != nullptr, "Cannot create bridge fence event");
    }
    void start(HMODULE addon, phoenix_model::WeightSource source,
               phoenix_addon::phoenix::Settings settings, FeedBuild build,
               std::optional<PortableOptions> portable = std::nullopt,
               std::optional<NativeOptions> native = std::nullopt) {
        // Only shared state is captured: destroying HostClient cannot leave a
        // worker referring to freed transport, fences, mutexes or UI buffers.
        portable_ = portable.has_value();
        native_=native.has_value();
        native_mode_=native&&native->driver_host?"in_process_x86_driver":"in_process_x64_library";
        auto self = shared_from_this();
        start_owned_worker(
            std::move(self),
            [addon, source = std::move(source), settings = std::move(settings),
             build, portable,native](HostState& state) {
                if(native) state.serve_native(source,settings,build,*native);
                else if (portable) state.serve_portable(source, settings, build, *portable);
                else state.serve(addon, source, settings, build);
            });
    }
    void stop() {
        stopping_ = true;
        changed_.notify_all();
    }
    DWORD process_id() const {
        return native_?GetCurrentProcessId():child_pid_.load();
    }
    Phase phase() const {
        return phase_.load();
    }
    FeedBuildAck fences() {
        require(phase() == Phase::OpenFences, "No shared fence handshake");
        return duplicate_fence_ack(process_.value, acknowledgement_);
    }
    void opened_fences(ID3D11Fence* output) {
        std::lock_guard lock(mutex_);
        output_fence_ = output;
        phase_ = Phase::Ready;
    }
    bool submit(uint64_t serial, bool reset, std::vector<uint8_t> pixels = {},
                uint32_t iterations = 1) {
        std::lock_guard lock(mutex_);
        if (phase() != Phase::Ready || iterations < 1 || iterations > 10)
            return false;
        request_ = {serial, uint32_t(reset), uint32_t(model_enabled_), iterations, 0};
        input_ = std::move(pixels);
        phase_ = Phase::Busy;
        changed_.notify_all();
        return true;
    }
    std::vector<uint8_t> take() {
        std::lock_guard lock(mutex_);
        require(phase() == Phase::Finished, "No completed graphics frame");
        auto result = std::move(output_);
        taken_tuning_revision_ = completed_tuning_revision_;
        phase_ = Phase::Ready;
        return result;
    }
    std::string error() {
        std::lock_guard lock(mutex_);
        return error_;
    }
    Json ui_snapshot() {
        std::lock_guard lock(mutex_);
        auto snapshot = ui_snapshot_;
        snapshot["live_tuning"] = {{"requested_revision", tuning_revision_},
            {"applied_revision", applied_tuning_revision_}, {"presented_revision", taken_tuning_revision_},
            {"pending", pending_tuning_.has_value()}};
        return snapshot;
    }
    bool update_tuning(const Settings& settings) {
        if (!settings.valid()) return false;
        std::lock_guard lock(mutex_);
        if (stopping_) return false;
        if (phase() == Phase::Failed || phase() == Phase::Stopped)
            return live_tuning_json(settings) == live_tuning_json(tuning_settings_);
        if (phoenix_addon::phoenix::copy_live_tuning(tuning_settings_, settings)) {
            pending_tuning_ = live_tuning_json(tuning_settings_).dump();
            ++tuning_revision_;
            changed_.notify_all();
        }
        return true;
    }
    bool tuning_current_output() {
        std::lock_guard lock(mutex_);
        return taken_tuning_revision_ == tuning_revision_;
    }
    bool ui_action(phoenix_ui::Action action, unsigned value = 0) {
        std::lock_guard lock(mutex_);
        if ((portable_||native_) && action != phoenix_ui::Action::ResetHistory &&
            action != phoenix_ui::Action::ClearTiming && action != phoenix_ui::Action::Marker)
            return false;
        if (!phoenix_ui::valid_action(uint32_t(action)) || phase() == Phase::Failed ||
            phase() == Phase::Stopped || ui_actions_.size() >= 16)
            return false;
        ui_actions_.push_back({uint32_t(action), value});
        changed_.notify_all();
        return true;
    }
    double roundtrip() {
        std::lock_guard lock(mutex_);
        return roundtrip_ms_;
    }
    uint64_t completed_at() {
        std::lock_guard lock(mutex_);
        return completed_at_;
    }
};

// A render-facing lease. Shutdown only requests cancellation; the worker owns
// its HostState until pipe I/O, helper termination and handle cleanup finish.
class HostClient {
    std::shared_ptr<HostState> state_;
    bool portable_ = false;
    std::optional<NativeOptions> native_;

  public:
    HostClient(HMODULE addon, const phoenix_model::WeightSource& source,
               const phoenix_addon::phoenix::Settings& settings, const FeedBuild& build,
               std::optional<PortableOptions> portable = std::nullopt,
               std::optional<NativeOptions> native = std::nullopt)
        : state_(std::make_shared<HostState>(settings)), portable_(portable.has_value()),native_(native) {
        state_->start(addon, source, settings, build, portable,native);
    }
    ~HostClient() {
        state_->stop();
    }
    DWORD process_id() const {
        return state_->process_id();
    }
    bool in_process_portable() const { return portable_; }
    bool in_process() const {return portable_||native_.has_value();}
    const char* native_execution() const {return !native_?nullptr:native_->driver_host?"in_process_x86_driver":"in_process_x64_library";}
    Phase phase() const {
        return state_->phase();
    }
    FeedBuildAck fences() {
        return state_->fences();
    }
    void opened_fences(ID3D11Fence* output) {
        state_->opened_fences(output);
    }
    bool submit(uint64_t serial, bool reset, std::vector<uint8_t> pixels = {},
                uint32_t iterations = 1) {
        return state_->submit(serial, reset, std::move(pixels), iterations);
    }
    std::vector<uint8_t> take() {
        return state_->take();
    }
    std::string error() {
        return state_->error();
    }
    Json ui_snapshot() {
        return state_->ui_snapshot();
    }
    bool update_tuning(const Settings& settings) { return state_->update_tuning(settings); }
    bool tuning_current_output() { return state_->tuning_current_output(); }
    bool ui_action(phoenix_ui::Action action, unsigned value = 0) {
        return state_->ui_action(action, value);
    }
    double roundtrip() {
        return state_->roundtrip();
    }
    uint64_t completed_at() {
        return state_->completed_at();
    }
};
} // namespace phoenix_compat
