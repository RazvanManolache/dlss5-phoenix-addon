// CPU-only C ABI fixture. Never loads a model or graphics driver.
#include "../../third_party/nlohmann/json.hpp"
#include <windows.h>
#include <string>
#include <atomic>
using Json = nlohmann::json;
struct Session { unsigned width, height, frames = 0; Json metadata; };
thread_local std::string failure;
std::atomic<bool> frame_active = false;
extern "C" __declspec(dllexport) int phoenix_test_frame_active() { return frame_active; }
extern "C" __declspec(dllexport) void* phoenix_runtime_create(const char*, unsigned w, unsigned h) {
    return new Session{w, h};
}
extern "C" __declspec(dllexport) void phoenix_runtime_destroy(void* value) { delete static_cast<Session*>(value); }
extern "C" __declspec(dllexport) int phoenix_runtime_frame(void* value, const float* color,
    const float* motion, const float* depth, const float* validity, const char* metadata, float* output) {
    auto& session = *static_cast<Session*>(value);
    session.metadata = Json::parse(metadata);
    if (motion || depth || validity || !session.metadata.at("reset").get<bool>()) return -1;
    frame_active = true;
    Sleep(80); // Makes cancellation/live-edit races reproducible without GPU work.
    frame_active = false;
    const float tone = session.metadata["controls"]["local_tone"];
    if (tone > 1.9f) { failure = "injected inference failure"; return -1; }
    for (size_t i = 0; i < size_t(session.width)*session.height*3; ++i)
        output[i] = std::clamp(color[i] * tone + .1f, 0.f, 1.f);
    ++session.frames;
    return 0;
}
extern "C" __declspec(dllexport) const char* phoenix_runtime_error() { return failure.c_str(); }
extern "C" __declspec(dllexport) size_t phoenix_runtime_status(void* value, char* out, size_t capacity) {
    const auto& session = *static_cast<Session*>(value);
    const auto text = Json{{"frames", session.frames}, {"metadata", session.metadata}}.dump();
    if (out && capacity >= text.size()+1) memcpy(out, text.c_str(), text.size()+1);
    return text.size()+1;
}
