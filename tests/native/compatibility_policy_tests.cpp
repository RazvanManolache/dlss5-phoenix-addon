#define NOMINMAX
#include "../../src/integration/compatibility/session_policy.h"
#include "../../src/native/support/control_mailbox.h"
#include <iostream>

int main() {
    unsigned checks = 0;
    auto verify = [&](bool condition) { ++checks; if (!condition) throw std::runtime_error("Policy assertion failed"); };
    namespace control = phoenix_addon::phoenix::control_bridge;
    try {
        phoenix_compat::Settings settings;
        settings.budget_mode = 1;
        verify(phoenix_compat::normalize_capabilities(settings));
        verify(settings.budget_mode == 0);
        verify(!phoenix_compat::normalize_capabilities(settings));
        auto changed = settings;
        changed.model_dll = "C:/another/model.dll";
        verify(phoenix_compat::helper_settings_changed(settings, changed));
        changed = settings;
        changed.offload_age_frames += 1;
        verify(phoenix_compat::helper_settings_changed(settings, changed));
        verify(!phoenix_compat::helper_settings_changed(settings, settings));

        const auto folder = std::filesystem::temp_directory_path() /
            (L"PhoenixMailboxTests-" + std::to_wstring(GetCurrentProcessId()));
        std::filesystem::create_directories(folder);
        struct Cleanup { std::filesystem::path folder; ~Cleanup() { std::error_code error; std::filesystem::remove_all(folder, error); } } cleanup{folder};
        control::Mailbox mailbox;
        mailbox.configure(folder, 123);
        bool enabled = false;
        unsigned commands = 0;
        auto snapshot = [&] { return nlohmann::json{{"enabled", enabled}}; };
        auto dispatch = [&](const control::Command& command, nlohmann::json&) {
            ++commands;
            if (command.enabled) enabled = *command.enabled;
            return "applied";
        };
        mailbox.poll(settings, snapshot, dispatch, 1000);
        verify(std::filesystem::is_regular_file(folder / "endpoint.json"));
        auto request = nlohmann::json{{"schema", "phoenix-control/1"}, {"session", mailbox.session()},
            {"pid", GetCurrentProcessId()}, {"id", "enable-one"}, {"op", "set"},
            {"enabled", true}, {"coherent_frames", false}, {"expires_tick_ms", 5000}};
        auto run = [&](const nlohmann::json& input, uint64_t tick) {
            { std::ofstream file(folder / "request.json"); file << input.dump(); }
            mailbox.poll(settings, snapshot, dispatch, tick);
            nlohmann::json response;
            std::ifstream(folder / "response.json") >> response;
            return response;
        };
        verify(run(request, 1300).at("ok") == true);
        verify(enabled && commands == 1);
        verify(run(request, 1600).at("ok") == false);
        verify(commands == 1);
        request["id"] = "invalid-bool";
        request["enabled"] = 1;
        verify(run(request, 1900).at("ok") == false);
        verify(commands == 1);
        request["enabled"] = false;
        request["id"] = "disable-two";
        verify(run(request, 2200).at("ok") == true);
        verify(!enabled && commands == 2);
        const auto original_id = request["id"];
        request["id"] = "enable-one";
        verify(run(request, 2600).at("ok") == false);
        verify(commands == 2);
        request["id"] = original_id;
        request["id"] = "expired";
        request["expires_tick_ms"] = 2300;
        verify(run(request, 2900).at("ok") == false);
        verify(commands == 2);
        request["id"] = "wrong-session";
        request["expires_tick_ms"] = 9000;
        request["session"] = "old";
        verify(run(request, 3200).at("ok") == false);
        verify(commands == 2);
        std::cout << nlohmann::json{{"passed", true}, {"cpu_only", true}, {"checks", checks}}.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
