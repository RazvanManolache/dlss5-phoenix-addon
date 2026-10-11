#pragma once
// JSONL writer for CU-kernel dispatch capture (c18g). Both our own launches
// (the phoenix executor calls NvAPI_D3D12_LaunchCuKernelChain) and the DLL's
// (native evaluate runs through the same hooked entry) land here with kernel
// names resolved from the CreateCuFunction hook, in launch order with full
// argument bytes -- the schedule/args alignment needs the constants, not
// just the names. CPU only: no MinHook, no NvAPI, no D3D12, so the unit
// tests pin the parser, the hex encoder and the exact line format. The
// interceptors themselves live in dispatch_hook.h and
// are integration-verified in-game (identical rendering with the hook on).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace phoenix_addon::dispatch_log {
// Frames cap from CNR_PHOENIX_DISPATCH_LOG. Absent, empty, unparseable or zero
// means capture off; huge values clamp so a typo can't spool unbounded
// output. Pure; the CPU tests pin each case.
constexpr unsigned max_log_frames = 100000u;
// MinHook status names for the install note (the note is the only diagnostic
// channel from inside the game, so failures must speak). Pure; tested.
inline const char* mh_status_name(int status) noexcept {
    switch (status) {
    case 0:
        return "ok"; // MH_OK
    case 1:
        return "already-initialized"; // MH_ERROR_ALREADY_INITIALIZED
    case 2:
        return "not-initialized"; // MH_ERROR_NOT_INITIALIZED
    case 3:
        return "already-created"; // MH_ERROR_ALREADY_CREATED
    case 4:
        return "not-created"; // MH_ERROR_NOT_CREATED
    case 5:
        return "enabled"; // MH_ERROR_ENABLED
    case 6:
        return "disabled"; // MH_ERROR_DISABLED
    case 7:
        return "not-executable"; // MH_ERROR_NOT_EXECUTABLE
    case 8:
        return "unsupported-function"; // MH_ERROR_UNSUPPORTED_FUNCTION
    case 9:
        return "memory-alloc"; // MH_ERROR_MEMORY_ALLOC
    case 10:
        return "memory-protect"; // MH_ERROR_MEMORY_PROTECT
    case 11:
        return "module-not-found"; // MH_ERROR_MODULE_NOT_FOUND
    case 12:
        return "function-not-found"; // MH_ERROR_FUNCTION_NOT_FOUND
    default:
        return "unknown";
    }
}
inline unsigned parse_log_frames(const char* text) noexcept {
    if (!text || !*text)
        return 0;
    // strtoull accepts a leading '-' by wrapping to huge, which would then
    // clamp to the max: a "-300" typo must disable, not spool 100k frames.
    const char* cursor = text;
    while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n' || *cursor == '\r')
        ++cursor;
    if (*cursor == '-')
        return 0;
    char* end = nullptr;
    const unsigned long long parsed_value = std::strtoull(text, &end, 10);
    if (end == text)
        return 0;
    if (parsed_value > max_log_frames)
        return max_log_frames;
    return static_cast<unsigned>(parsed_value);
}
// Lowercase hex, no prefix; null/empty input encodes empty.
inline std::string hex_bytes(const void* data, size_t size) {
    static const char digits[] = "0123456789abcdef";
    if (!data || !size)
        return {};
    std::string out;
    out.resize(size * 2);
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t iteration_index = 0; iteration_index < size; ++iteration_index) {
        out[iteration_index * 2] = digits[bytes[iteration_index] >> 4];
        out[iteration_index * 2 + 1] = digits[bytes[iteration_index] & 15];
    }
    return out;
}
// Minimal JSON string escaping for kernel names (normally [a-z0-9_]).
inline std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 2);
    for (const char character : text) {
        const auto byte = static_cast<unsigned char>(character);
        if (character == '\\')
            out += "\\\\";
        else if (character == '"')
            out += "\\\"";
        else if (character == '\n')
            out += "\\n";
        else if (character == '\r')
            out += "\\r";
        else if (character == '\t')
            out += "\\t";
        else if (byte < 0x20) {
            char escape_buffer[8]{};
            std::snprintf(escape_buffer, sizeof(escape_buffer), "\\u%04x", unsigned(byte));
            out += escape_buffer;
        } else
            out += character;
    }
    return out;
}
// Argument bytes beyond this are cut with "trunc":true. Real launches carry
// tens to hundreds of bytes; the cap is pure safety against a wild size.
constexpr size_t max_logged_params = 4096;
// Exact line format, golden-pinned by the CPU tests. `side` is 0 for the
// DLL's launches, 1 for ours. `function` prints as a hex string (a JSON
// number cannot hold 64-bit handles exactly).
inline std::string format_launch_line(uint64_t frame, unsigned side, uint64_t seq, const char* name,
                                      uint64_t function, const unsigned grid[3],
                                      const unsigned block[3], unsigned shared, unsigned param_size,
                                      const void* params) {
    const size_t logged = param_size > max_logged_params ? max_logged_params : size_t(param_size);
    char callback[24]{};
    std::snprintf(callback, sizeof(callback), "0x%llx", static_cast<unsigned long long>(function));
    std::string line = "{\"f\":" + std::to_string(frame) + ",\"side\":\"" +
                       (side ? "ours" : "dll") + "\",\"seq\":" + std::to_string(seq) +
                       ",\"name\":\"" + json_escape(name ? name : "unknown") + "\",\"fn\":\"" +
                       callback + "\",\"grid\":[" + std::to_string(grid[0]) + "," +
                       std::to_string(grid[1]) + "," + std::to_string(grid[2]) + "],\"block\":[" +
                       std::to_string(block[0]) + "," + std::to_string(block[1]) + "," +
                       std::to_string(block[2]) + "],\"shared\":" + std::to_string(shared) +
                       ",\"params\":" + std::to_string(param_size) + ",\"args\":\"" +
                       hex_bytes(params, logged) + "\"";
    if (param_size > max_logged_params)
        line += ",\"trunc\":true";
    line += "}\n";
    return line;
}
// Append-only, mutex-guarded (native evaluate runs on several threads),
// unbuffered so a game crash keeps everything written so far. Best-effort:
// disk-full shortfalls are silent, like the trace writer.
class Log {
  public:
    Log() = default;
    Log(const Log&) = delete;
    Log& operator=(const Log&) = delete;
    ~Log() {
        close();
    }
    bool open(const char* path, unsigned max_frames) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!path || !*path || !max_frames || file_)
            return false;
        FILE* file = nullptr;
        if (fopen_s(&file, path, "wb") != 0 || !file)
            return false;
        std::setvbuf(file, nullptr, _IONBF, 0);
        file_ = file;
        max_frames_ = max_frames;
        return true;
    }
    // Appends one launch with its per-(frame,side) sequence number. Silent
    // no-op when closed or once frame reaches the cap.
    void launch(uint64_t frame, unsigned side, const char* name, uint64_t function,
                const unsigned grid[3], const unsigned block[3], unsigned shared,
                unsigned param_size, const void* params) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!file_ || frame >= max_frames_ || side > 1)
            return;
        if (frame != frame_ || side != side_) {
            frame_ = frame;
            side_ = side;
            seq_ = 0;
        }
        const std::string line = format_launch_line(frame, side, seq_++, name, function, grid,
                                                    block, shared, param_size, params);
        std::fwrite(line.data(), 1, line.size(), file_);
        if (frame >= frames_)
            frames_ = frame + 1;
        ++launches_;
    }
    uint64_t frames() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return frames_;
    }
    uint64_t launches() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return launches_;
    }
    void close() noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            if (file_) {
                std::fclose(file_);
                file_ = nullptr;
            }
        } catch (...) {
        }
    }

  private:
    mutable std::mutex mutex_;
    FILE* file_ = nullptr;
    unsigned max_frames_ = 0;
    uint64_t frames_ = 0, frame_ = 0, seq_ = 0, launches_ = 0;
    unsigned side_ = 0;
};
} // namespace production_game41::dispatch_log
