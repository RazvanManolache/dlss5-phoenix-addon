#pragma once
// Trace manifest assembly, dependency-free so the CPU gates can pin it.
//
// The manifest is the trace's cover page: the replay tool reads the role
// geometry from it, and `frames` MUST equal the frames actually spooled to
// frames.jsonl (a quit-before-drain leaves submitted-but-uncompleted work,
// which is reported separately as `submitted`, never as frames).
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace phoenix_addon::trace {
struct ManifestRole {
    std::string name;
    bool known = false;
    unsigned format = 0;
    uint64_t width = 0, height = 0, row_pitch = 0, num_rows = 0, total_bytes = 0;
};

inline std::string build_manifest_json(const char* utc, uint64_t frames, uint64_t submitted,
                                       uint64_t dropped, uint64_t gaps,
                                       const std::vector<ManifestRole>& roles,
                                       const std::string& note) {
    std::ostringstream manifest;
    manifest << "{\"schema\":\"production-game-phoenix-trace/1\",\"utc\":\"" << utc << "\""
             << ",\"frames\":" << frames << ",\"submitted\":" << submitted
             << ",\"dropped\":" << dropped << ",\"gaps\":" << gaps << ",\"roles\":[";
    for (size_t role_index = 0; role_index < roles.size(); ++role_index) {
        const auto& group = roles[role_index];
        manifest << (role_index ? "," : "") << "{\"name\":\"" << group.name
                 << "\",\"known\":" << (group.known ? "true" : "false");
        if (group.known) {
            manifest << ",\"format\":" << group.format << ",\"width\":" << group.width
                     << ",\"height\":" << group.height << ",\"row_pitch\":" << group.row_pitch
                     << ",\"num_rows\":" << group.num_rows
                     << ",\"total_bytes\":" << group.total_bytes;
        }
        manifest << "}";
    }
    manifest << "],\"note\":\"" << note << "\"}\n";
    return manifest.str();
}
} // namespace production_game41::trace
