#pragma once
#include "control_bridge.h"
#include <fstream>
#include <map>
#include <filesystem>
#include <windows.h>

namespace phoenix_addon::phoenix::control_bridge {
// A bounded local control mailbox. The adapter executes commands on its game
// runtime callback; no UI pointers or GPU objects cross this file boundary.
class Mailbox {
    std::filesystem::path folder_;
    std::string session_;
    std::map<std::string, uint64_t> seen_;
    uint64_t poll_tick_ = 0;
    bool published_ = false;
    Json pending_;
    static void write(const std::filesystem::path& path, const Json& value) {
        auto temporary = path;
        temporary += L".tmp";
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        check(bool(output), "Control response cannot be opened");
        output << value.dump(2);
        output.flush();
        check(bool(output), "Control response write failed");
        output.close();
        check(MoveFileExW(temporary.c_str(), path.c_str(),
                          MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0,
              "Control response publication failed");
    }
  public:
    const std::filesystem::path& folder() const { return folder_; }
    const std::string& session() const { return session_; }
    void configure(std::filesystem::path folder, uint64_t window) {
        folder_ = std::move(folder);
        FILETIME created{}, exited{}, kernel{}, user{};
        check(GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user) != 0,
              "Control process identity unavailable");
        session_ = std::to_string(GetCurrentProcessId()) + "-" +
                   std::to_string((uint64_t(created.dwHighDateTime) << 32) | created.dwLowDateTime) +
                   "-" + std::to_string(window) + "-" + std::to_string(GetTickCount64());
    }
    template<class Snapshot, class Dispatch>
    void poll(const Settings& settings, Snapshot snapshot, Dispatch dispatch,
              uint64_t now = GetTickCount64()) {
        if (folder_.empty() || now - poll_tick_ < 250) return;
        poll_tick_ = now;
        if (!published_) {
            std::filesystem::create_directories(folder_);
            write(folder_ / "endpoint.json", {{"schema", "phoenix-control/1"},
                {"session", session_}, {"pid", GetCurrentProcessId()},
                {"frontend_controls", {"enabled", "coherent_frames"}},
                {"operations", {"status", "set", "save", "reset_history", "screenshot", "screenshot_pair", "capture", "validate_model"}}});
            published_ = true;
        }
        if (!pending_.is_null()) {
            write(folder_ / "response.json", pending_);
            pending_ = nullptr;
        }
        const auto incoming = folder_ / "request.json", claimed = folder_ / "processing.json";
        if (!std::filesystem::exists(incoming)) return;
        check(MoveFileExW(incoming.c_str(), claimed.c_str(), MOVEFILE_REPLACE_EXISTING) != 0,
              "Control request claim failed");
        Json response{{"schema", "phoenix-control/1"}, {"session", session_},
                      {"pid", GetCurrentProcessId()}, {"id", nullptr}, {"ok", false}};
        try {
            check(std::filesystem::file_size(claimed) <= 65536, "Control request exceeds 64 KiB");
            std::ifstream input(claimed, std::ios::binary);
            std::string bytes(65537, '\0');
            input.read(bytes.data(), std::streamsize(bytes.size()));
            bytes.resize(size_t(input.gcount()));
            check(bytes.size() <= 65536, "Control request exceeds 64 KiB");
            const auto request = Json::parse(bytes);
            if (request.contains("id") && request["id"].is_string() &&
                request["id"].get_ref<const std::string&>().size() <= 64) response["id"] = request["id"];
            const auto command = validate(request, session_, GetCurrentProcessId(), now, settings);
            for (auto entry = seen_.begin(); entry != seen_.end();) {
                if (entry->second <= now) entry = seen_.erase(entry);
                else ++entry;
            }
            check(!seen_.contains(command.id), "Duplicate control request; inspect previous response");
            check(seen_.size() < 128, "Control request window is full; retry after expiry");
            seen_[command.id] = request.at("expires_tick_ms").get<uint64_t>();
            response["phase"] = dispatch(command, response);
            response["ok"] = true;
        } catch (const std::exception& error) {
            response["phase"] = "rejected";
            response["error"] = error.what();
        }
        response["snapshot"] = snapshot();
        pending_ = std::move(response);
        std::error_code error;
        std::filesystem::remove(claimed, error);
        write(folder_ / "response.json", pending_);
        pending_ = nullptr;
    }
};
} // namespace phoenix_addon::phoenix::control_bridge
