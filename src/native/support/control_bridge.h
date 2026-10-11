#pragma once
#include "overrides.h"
#include "../session/session.h"

namespace phoenix_addon::phoenix::control_bridge {
// Local bounded mailbox. Pure validation is shared by the addon and CPU tests.
struct Command {
    std::string id, op;
    Settings settings;
    unsigned frames = 0;
};
inline Command validate(const Json& request, const std::string& session, unsigned pid, uint64_t now,
                        const Settings& current) {
    check(request.is_object(), "Bridge request must be an object");
    for (auto lookup_iterator = request.begin(); lookup_iterator != request.end();
         ++lookup_iterator)
        check(lookup_iterator.key() == "schema" || lookup_iterator.key() == "session" ||
                  lookup_iterator.key() == "pid" || lookup_iterator.key() == "expires_tick_ms" ||
                  lookup_iterator.key() == "id" || lookup_iterator.key() == "op" ||
                  lookup_iterator.key() == "settings" || lookup_iterator.key() == "frames",
              "Unknown bridge request field");
    check(request.at("schema") == "phoenix-control/1", "Bridge schema mismatch");
    check(request.at("session") == session && request.at("pid") == pid,
          "Bridge session changed; reconnect");
    const auto& expiry = request.at("expires_tick_ms");
    check(expiry.is_number_unsigned() || (expiry.is_number_integer() && expiry.get<int64_t>() >= 0),
          "Invalid bridge expiry");
    const auto deadline = expiry.get<uint64_t>();
    check(deadline > now && deadline - now <= 30000, "Bridge request expired or deadline too long");
    Command cmd;
    cmd.id = request.at("id").get<std::string>();
    cmd.op = request.at("op").get<std::string>();
    check(!cmd.id.empty() && cmd.id.size() <= 64, "Invalid bridge request id");
    for (unsigned char character : cmd.id)
        check((character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
                  (character >= '0' && character <= '9') || character == '-' || character == '_',
              "Invalid bridge request id character");
    check(cmd.op == "status" || cmd.op == "set" || cmd.op == "reset_history" ||
              cmd.op == "screenshot" || cmd.op == "capture" || cmd.op == "save" ||
              cmd.op == "validate_model",
          "Unknown bridge operation");
    cmd.settings = current;
    check((cmd.op == "set") == request.contains("settings"), "Settings only allowed for set");
    check((cmd.op == "capture") == request.contains("frames"), "Frames only allowed for capture");
    if (cmd.op == "set") {
        const auto& values = request.at("settings");
        check(values.is_object() && !values.empty() && values.size() <= 128,
              "Invalid bridge settings object");
        for (auto lookup_iterator = values.begin(); lookup_iterator != values.end();
             ++lookup_iterator) {
            check(lookup_iterator.key() != "ScreenshotRequest" &&
                      lookup_iterator.key() != "CaptureBurst",
                  "Use explicit capture operations");
            bool found = false;
            visit_settings(cmd.settings, [&](const char* key, auto& field) {
                if (lookup_iterator.key() != key)
                    return;
                found = true;
                using T = std::decay_t<decltype(field)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    check(lookup_iterator.value().is_string(), "String setting required");
                    field = lookup_iterator.value().get<std::string>();
                    return;
                } else if constexpr (std::is_same_v<T, bool>)
                    check(lookup_iterator.value().is_boolean(), "Boolean setting required");
                else if constexpr (std::is_integral_v<T>)
                    check(lookup_iterator.value().is_number_integer(), "Integer setting required");
                else
                    check(lookup_iterator.value().is_number(), "Numeric setting required");
                check(parse_override(lookup_iterator.value().dump(), field),
                      "Setting is out of range");
            });
            check(found, "Unknown bridge setting");
        }
        check(cmd.settings.valid(), "Bridge settings rejected; current settings retained");
    }
    if (cmd.op == "capture") {
        check(request.at("frames").is_number_integer(), "Integer capture count required");
        const auto frame_count = request.at("frames").get<int64_t>();
        check(frame_count >= 1 && frame_count <= 8, "Bridge capture count must be 1..8");
        cmd.frames = unsigned(frame_count);
    }
    return cmd;
}
}
