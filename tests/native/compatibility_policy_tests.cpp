#define NOMINMAX
#include "../../src/integration/compatibility/session_policy.h"
#include "../../src/integration/compatibility/portable_policy.h"
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
        changed.nvidia_dll_folder = "C:/shared-library";
        verify(phoenix_compat::helper_settings_changed(settings, changed));
        verify(phoenix_compat::settings_json(changed).at("NvidiaDllFolder") == "C:/shared-library");
        verify(!phoenix_compat::live_tuning_json(changed).contains("NvidiaDllFolder"));
        changed = settings;
        changed.offload_age_frames += 1;
        verify(phoenix_compat::helper_settings_changed(settings, changed));
        verify(!phoenix_compat::helper_settings_changed(settings, settings));
        auto tuned = settings;
        tuned.controls.enabled = tuned.controls.advanced = tuned.controls.auto_mask = true;
        tuned.controls.tone = .4f; tuned.controls.structure = 1.8f; tuned.controls.skin = .3f;
        tuned.native_options = true; tuned.intensity = .6f;
        tuned.style = 2; tuned.host_exposure = 1.3f;
        tuned.model_dll = "pending-model.dll"; tuned.resolution_mode = 1;
        tuned.nvidia_dll_folder = "C:/pending-library";
        auto hot = settings;
        verify(phoenix_addon::phoenix::copy_live_tuning(hot, tuned));
        verify(hot.controls.skin == .3f && hot.controls.auto_mask && hot.controls.tone == .4f);
        verify(hot.intensity == .6f && hot.style == 2 && hot.host_exposure == 1.3f);
        verify(hot.model_dll == settings.model_dll && hot.resolution_mode == settings.resolution_mode);
        verify(hot.nvidia_dll_folder == settings.nvidia_dll_folder);
        verify(!phoenix_compat::helper_settings_changed(settings, hot));
        verify(phoenix_compat::helper_settings_changed(hot, tuned));
        verify(!phoenix_addon::phoenix::copy_live_tuning(hot, tuned));
        auto patch = phoenix_compat::live_tuning_json(hot);
        auto decoded = settings;
        phoenix_compat::apply_live_tuning_patch(decoded, patch);
        verify(phoenix_compat::live_tuning_json(decoded) == patch);
        verify(decoded.model_dll == settings.model_dll && decoded.resolution_mode == settings.resolution_mode);
        for (auto invalid : {nlohmann::json{{"Tone", 3}}, nlohmann::json{{"Tone", "1"}},
                             nlohmann::json{{"AutoMask", 1}}, nlohmann::json{{"Depth", 4294967296ull}},
                             nlohmann::json{{"ModelDll", "other.dll"}},
                             nlohmann::json{{"NvidiaDllFolder", "other-folder"}},
                             nlohmann::json{{"Tone", .8}, {"Skin", 100}}}) {
            const auto before = phoenix_compat::settings_json(decoded);
            bool refused = false;
            try { phoenix_compat::apply_live_tuning_patch(decoded, invalid); }
            catch (const std::exception&) { refused = true; }
            verify(refused && phoenix_compat::settings_json(decoded) == before);
        }
        for (int mode = 0; mode != 4; ++mode) {
            changed.mode = mode;
            verify(phoenix_compat::helper_presentation_mode(changed) == (mode == 2 ? 3 : mode));
        }
        changed.mode = 2;
        for (int view = 1; view <= 5; ++view) {
            changed.comparison = view;
            verify(phoenix_compat::helper_presentation_mode(changed) == 3);
        }
        for (bool bridge_enabled : {false, true})
            for (bool nr_enabled : {false, true}) {
                auto migrated = settings;
                migrated.uplift = nr_enabled;
                bool enabled = bridge_enabled;
                phoenix_compat::migrate_processing_switch(enabled, migrated);
                verify(enabled == (bridge_enabled && nr_enabled));
                verify(migrated.uplift);
            }
        auto draft = settings;
        auto applied = settings;
        draft.model_dll = "C:/pending/model.dll";
        draft.controls.tone = 1.7f;
        draft.resolution_mode = 1;
        draft.uplift = false;
        for (int mode : {2, 1, 3, 0, 3, 2}) {
            draft.mode = mode;
            draft.comparison = 1;
            draft.source_a = 0;
            draft.source_b = 2;
            draft.divider = .42f;
            verify(phoenix_compat::apply_live_selection(applied, draft));
            verify(applied.mode == mode && applied.comparison == 1 && applied.divider == .42f);
            verify(applied.source_a == 0 && applied.source_b == 2 && applied.uplift);
            verify(applied.model_dll == settings.model_dll && applied.controls.tone == settings.controls.tone);
            verify(applied.resolution_mode == settings.resolution_mode);
            verify(!phoenix_compat::apply_live_selection(applied, draft));
        }
        std::swap(draft.source_a, draft.source_b);
        verify(phoenix_compat::apply_live_selection(applied, draft));
        verify(applied.source_a == 2 && applied.source_b == 0);
        auto pending = draft;
        applied.controls.tone = .7f;
        phoenix_compat::merge_settings_patch(pending, applied, {});
        verify(pending.model_dll == draft.model_dll && pending.controls.tone == draft.controls.tone);
        phoenix_compat::merge_settings_patch(pending, applied, {"Tone"});
        verify(pending.controls.tone == .7f && pending.model_dll == draft.model_dll);
        phoenix_compat::DeferredTimeout timeout;
        verify(!timeout.expired(100));
        verify(!timeout.expired(10099));
        verify(timeout.expired(10100));
        timeout.reset(); // Input/off suspends the previous attempt.
        verify(!timeout.expired(60000));
        verify(!timeout.expired(69999));
        timeout.reset(); // A new selection or Bypass starts a fresh attempt.
        verify(!timeout.expired(80000));
        verify(timeout.expired(90000));
        verify(phoenix_compat::backbuffer_requires_reset(false, false));
        verify(phoenix_compat::backbuffer_requires_reset(false, true));
        verify(!phoenix_compat::backbuffer_requires_reset(true, false));
        verify(phoenix_compat::backbuffer_requires_reset(true, true));

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
        for(const char* model : {"original", "small-fp8"}) {
            phoenix_compat::Settings portable;
            portable.model_id = model;
            portable.mode = phoenix_addon::phoenix::ours;
            portable.experimental = portable.automatic_budget = true;
            portable.transformer_blocks = 5;
            portable.minimum_blocks = 2;
            portable.budget_ms = 3.5f;
            portable.block_repeats = {0,2,1,0,4,1,2,1};
            const auto metadata = phoenix_compat::portable_metadata(portable,true);
            verify(metadata.at("adaptive_budget").at("target_ms") == 3.5);
            verify(metadata.at("adaptive_budget").at("minimum_blocks") == 2);
            verify(metadata.at("transformer_blocks") == 5 && metadata.at("repeats").at(6) == 2);
            portable.budget_mode = 1;
            bool rejected = false;
            try {phoenix_compat::portable_metadata(portable,true);} catch(const std::exception&) {rejected=true;}
            verify(rejected);
        }
        std::cout << nlohmann::json{{"passed", true}, {"cpu_only", true}, {"checks", checks}}.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
