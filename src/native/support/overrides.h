#pragma once
#include "routing.h"
#include <sstream>
#include <locale>
#include <type_traits>

namespace phoenix_addon::phoenix {
inline std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}
template <class T> bool parse_override(const std::string& text, T& output) {
    if constexpr (std::is_same_v<T, bool>) {
        if (text == "1" || text == "true") {
            output = true;
            return true;
        }
        if (text == "0" || text == "false") {
            output = false;
            return true;
        }
        return false;
    } else {
        std::istringstream stream(text);
        stream.imbue(std::locale::classic());
        T parsed{};
        if (!(stream >> parsed))
            return false;
        stream >> std::ws;
        if (!stream.eof())
            return false;
        output = parsed;
        return true;
    }
}
// Partial edits preserve omitted keys. Invalid values are ignored; cross-field
// invalidity rejects the transaction rather than leaving an unsafe combination.
inline bool merge_overrides(const std::string& text, Settings& settings, unsigned& applied,
                            unsigned& ignored) {
    applied = ignored = 0;
    if (text.size() > 65536)
        return false;
    auto next = settings;
    std::istringstream stream(text);
    std::string line;
    bool section = false;
    while (std::getline(stream, line)) {
        line = trim(line);
        if (line.empty() || line[0] == ';' || line[0] == '#')
            continue;
        if (line[0] == '[') {
            section = line == "[Phoenix]";
            continue;
        }
        if (!section)
            continue;
        const auto equal = line.find('=');
        if (equal == std::string::npos) {
            ++ignored;
            continue;
        }
        const auto key = trim(line.substr(0, equal)), value = trim(line.substr(equal + 1));
        bool found = false, parsed = false;
        visit_settings(next, [&](const char* name, auto& field) {
            if (key == name) {
                found = true;
                parsed = parse_override(value, field);
            }
        });
        if (found && parsed)
            ++applied;
        else
            ++ignored;
    }
    if (!next.valid())
        return false;
    settings = next;
    return true;
}
}
