// Native file/pipe frontend. Human diagnostics use stderr; stdout is reserved
// for framed binary output in streaming mode. No shell or Python is launched.
#include "phoenix_native.h"
#include "../third_party/nlohmann/json.hpp"
#include <windows.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <cstring>

using Json = nlohmann::json;
namespace {
#pragma pack(push, 1)
struct Header {
    char magic[4]{'P','X','F','1'};
    uint16_t version = 1, bytes = 60;
    uint32_t width = 0, height = 0;
    uint64_t stream = 0, sequence = 0;
    int64_t timestamp_ns = 0;
    uint64_t color_bytes = 0, motion_bytes = 0;
    uint32_t flags = 0; // bit 0: reset, bit 1: engine motion valid
};
#pragma pack(pop)
static_assert(sizeof(Header) == 60);
struct File {
    HANDLE handle = INVALID_HANDLE_VALUE;
    bool owned = false, complete = false;
    std::filesystem::path output;
    ~File() {
        if (owned && handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
        if (!complete && !output.empty()) DeleteFileW(output.c_str());
    }
    void input(const std::filesystem::path& path) {
        if (path.empty()) handle = GetStdHandle(STD_INPUT_HANDLE);
        else { handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr); owned = true; }
        if (!handle || handle == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open input");
    }
    void destination(const std::filesystem::path& path) {
        if (path.empty()) handle = GetStdHandle(STD_OUTPUT_HANDLE);
        else {
            handle = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
            owned = true;
            if (handle != INVALID_HANDLE_VALUE) output = path;
        }
        if (!handle || handle == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot create output; existing files are preserved");
    }
    bool read(void* data, size_t bytes, bool allow_eof = false) {
        size_t offset = 0;
        while (offset != bytes) {
            DWORD count = 0;
            if (!ReadFile(handle, static_cast<char*>(data) + offset, DWORD(std::min<size_t>(bytes - offset, 1u << 26)), &count, nullptr)) {
                if (GetLastError() == ERROR_BROKEN_PIPE && !offset && allow_eof) return false;
                throw std::runtime_error("Input read failed");
            }
            if (!count) {
                if (!offset && allow_eof) return false;
                throw std::runtime_error("Truncated frame or frame header");
            }
            offset += count;
        }
        return true;
    }
    void write(const void* data, size_t bytes) {
        size_t offset = 0;
        while (offset != bytes) {
            DWORD count = 0;
            if (!WriteFile(handle, static_cast<const char*>(data) + offset, DWORD(std::min<size_t>(bytes - offset, 1u << 26)), &count, nullptr) || !count)
                throw std::runtime_error("Output write failed");
            offset += count;
        }
    }
};
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void native_ok(int result) { if (result) throw std::runtime_error(phoenix_native_error()); }
Json status(PhoenixNativeSession* session) {
    std::string text(phoenix_native_status(session, nullptr, 0), '\0');
    require(!text.empty() && phoenix_native_status(session, text.data(), text.size()) == text.size(), "Cannot read native status");
    return Json::parse(text.c_str());
}
Json config_file(const std::filesystem::path& path) {
    require(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) <= 256 * 1024,
        "Configuration must be a JSON file smaller than 256 KiB");
    std::ifstream file(path, std::ios::binary); return Json::parse(file);
}
}
int wmain(int argc, wchar_t** argv) {
    try {
        std::filesystem::path config_path, input_path, output_path, receipt_path;
        bool stream = false;
        for (int i = 1; i < argc; ++i) {
            const std::wstring argument = argv[i];
            if (argument == L"--help" || argument == L"-h") {
                std::cout << "phoenix-process --config recipe.json --input frame.rgba16f --output result.rgba16f\n"
                    "phoenix-process --config recipe.json --stream [--input framed.bin] [--output framed.bin]\n"
                    "--receipt report.json writes the completed job and runtime identity.\n"
                    "Stream: little-endian PXF1 version 1, 60-byte headers; RGB+A IEEE half pixels and optional RG half motion.\n";
                return 0;
            }
            if (argument == L"--capabilities") {
                std::cout << Json{{"schema", "phoenix-cli/1"}, {"native", true}, {"python_required", false},
                    {"models", {"original", "small-fp8"}}, {"frame_format", "rgba16f"},
                    {"stream_protocol", "PXF1/1"}, {"native_frame_info_version", 1},
                    {"backends", {"native-rtx", "spatial-d3d12"}}}.dump() << '\n';
                return 0;
            }
            if (argument == L"--stream") { stream = true; continue; }
            require(i + 1 < argc, "Missing argument value");
            if (argument == L"--config") config_path = argv[++i];
            else if (argument == L"--input") input_path = argv[++i];
            else if (argument == L"--output") output_path = argv[++i];
            else if (argument == L"--receipt") receipt_path = argv[++i];
            else throw std::runtime_error("Unknown command-line option; use --help");
        }
        require(!config_path.empty(), "Provide --config");
        require(stream || (!input_path.empty() && !output_path.empty()), "Single frames require --input and --output");
        auto config = config_file(config_path);
        const auto width = config.at("width").get<uint32_t>(), height = config.at("height").get<uint32_t>();
        const auto output_width = config.value("output_width", width), output_height = config.value("output_height", height);
        const auto color_bytes = uint64_t(width) * height * 8, output_bytes = uint64_t(output_width) * output_height * 8;
        require(width && height && output_width && output_height && width <= 16384 && height <= 16384 &&
            output_width <= 16384 && output_height <= 16384 && color_bytes <= (1ull << 31) && output_bytes <= (1ull << 31),
            "Invalid or oversized frame dimensions");
        if (!output_path.empty()) require(!std::filesystem::exists(output_path), "Output already exists");
        if (!receipt_path.empty()) require(!std::filesystem::exists(receipt_path), "Receipt already exists");
        config["offline"]=true;
        auto* session = phoenix_native_create(config.dump().c_str(), nullptr, nullptr);
        require(session != nullptr, phoenix_native_error());
        struct SessionGuard { PhoenixNativeSession* session; ~SessionGuard() { phoenix_native_destroy(session); } } guard{session};
        File input, output; input.input(input_path); output.destination(output_path);
        std::vector<uint16_t> color(size_t(color_bytes / 2)), result(size_t(output_bytes / 2)), motion;
        uint64_t frames = 0;
        while (true) {
            Header header; header.width = width; header.height = height; header.color_bytes = color_bytes;
            header.sequence = frames; header.timestamp_ns = int64_t(frames) * 16666667; header.flags = frames ? 0 : 1;
            if (stream) {
                if (!input.read(&header, sizeof(header), true)) break;
                require(!std::memcmp(header.magic, "PXF1", 4) && header.version == 1 && header.bytes == sizeof(header) &&
                    header.width == width && header.height == height && header.color_bytes == color_bytes &&
                    (header.motion_bytes == 0 || header.motion_bytes == uint64_t(width) * height * 4) &&
                    !(header.flags & ~3u) && (!(header.flags & 2u) || header.motion_bytes), "Invalid PXF1 frame header");
            }
            input.read(color.data(), size_t(color_bytes));
            motion.resize(size_t(header.motion_bytes / 2));
            if (!motion.empty()) input.read(motion.data(), size_t(header.motion_bytes));
            PhoenixNativeFrameInfo identity{sizeof(identity), 1, header.stream, header.sequence, header.timestamp_ns,
                uint32_t((header.flags & 2u) != 0), 0};
            native_ok(phoenix_native_set_frame_info(session, &identity));
            PhoenixNativeControls controls; phoenix_native_default_controls(&controls); controls.reset = header.flags & 1u;
            native_ok(phoenix_native_frame_host(session, color.data(), size_t(color_bytes), width * 8,
                motion.empty() ? nullptr : motion.data(), size_t(header.motion_bytes), width * 4,
                result.data(), size_t(output_bytes), output_width * 8, &controls));
            if (stream) {
                header.width = output_width; header.height = output_height; header.color_bytes = output_bytes;
                header.motion_bytes = 0; header.flags &= 1u; output.write(&header, sizeof(header));
            }
            output.write(result.data(), size_t(output_bytes)); ++frames;
            if (!stream) break;
        }
        require(frames != 0, "No input frames");
        const Json report{{"schema", "phoenix-cli-job/1"}, {"completed", true}, {"frames", frames},
            {"streamed", stream}, {"runtime", status(session)}};
        if (!receipt_path.empty()) {
            File receipt; receipt.destination(receipt_path);
            const auto text = report.dump(2) + '\n'; receipt.write(text.data(), text.size()); receipt.complete = true;
        }
        output.complete = true;
        std::cerr << report.dump() << '\n'; return 0;
    } catch (const std::exception& error) {
        std::cerr << Json{{"completed", false}, {"error", error.what()}}.dump() << '\n'; return 1;
    }
}
