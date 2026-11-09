#define PHOENIX_RUNTIME_IMPLEMENTATION
#include "phoenix.h"
#include "phoenix_linux.h"
#include "native_cuda_executor.h"
#include "frame_planes.h"
#include <mutex>

namespace {
using Json = nlohmann::json;
thread_local std::string last_error;
struct Session {
    std::mutex mutex;
    phoenix_native_cuda::Executor executor;
    Session(const Json& config, unsigned width, unsigned height)
        : executor(config, width, height) {}
};
size_t json_result(const Json& value, char* destination, size_t capacity) {
    const auto text = value.dump();
    const size_t bytes = text.size() + 1;
    if (destination && capacity >= bytes)
        std::memcpy(destination, text.c_str(), bytes);
    return bytes;
}
int unsupported(const char* message) {
    last_error = message;
    return -1;
}
void absent(const phoenix_plane_f32& plane) {
    if (plane.data || plane.capacity_bytes || plane.row_stride_bytes || plane.width ||
        plane.height || plane.channels || plane.reserved)
        throw std::runtime_error("Native CUDA reset-RGB contract requires absent optional planes");
}
} // namespace
extern "C" void* phoenix_runtime_create(const char* configuration, unsigned width,
                                        unsigned height) {
    try {
        if (!configuration)
            throw std::runtime_error("Missing native CUDA configuration");
        auto config = Json::parse(configuration);
        last_error.clear();
        return new Session(config, width, height);
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}
extern "C" void phoenix_runtime_destroy(void* session) { delete static_cast<Session*>(session); }
extern "C" const char* phoenix_runtime_error() { return last_error.c_str(); }
extern "C" int phoenix_runtime_frame_v2(void* handle, const phoenix_frame_v2* frame) {
    try {
        if (!handle || !frame || frame->struct_bytes != sizeof(*frame) ||
            frame->version != PHOENIX_FRAME_VERSION_2 || frame->reserved ||
            frame->control_profile != PHOENIX_CONTROL_NONE)
            throw std::runtime_error("Invalid native CUDA V2 frame header");
        auto& session = *static_cast<Session*>(handle);
        std::lock_guard guard(session.mutex);
        const auto width = session.executor.width, height = session.executor.height;
        const size_t stride = phoenix_cpp::plane_stride(frame->output, width, height, 3);
        absent(frame->motion);
        absent(frame->depth);
        absent(frame->validity);
        absent(frame->control_mask);
        const auto metadata =
            frame->metadata_json ? Json::parse(frame->metadata_json) : Json::object();
        if (!metadata.is_object() || metadata.size() != 1 || !metadata.value("reset", false))
            throw std::runtime_error("Native CUDA preview requires exactly metadata {reset:true}");
        // Snapshot before any execution/output write, including aliased planes.
        const auto color = phoenix_cpp::copy_plane(frame->color, width, height, 3, true);
        const auto& output = session.executor.run(color.data());
        auto* destination = reinterpret_cast<uint8_t*>(frame->output.data);
        for (unsigned y = 0; y < height; ++y)
            std::memcpy(destination + size_t(y) * stride, output.data() + size_t(y) * width * 3,
                        size_t(width) * 3 * sizeof(float));
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}
extern "C" size_t phoenix_runtime_status(void* handle, char* destination, size_t capacity) {
    try {
        if (!handle)
            throw std::runtime_error("Missing native CUDA session");
        auto& session = *static_cast<Session*>(handle);
        std::lock_guard guard(session.mutex);
        return json_result(session.executor.status, destination, capacity);
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}
extern "C" size_t phoenix_runtime_capabilities(char* destination, size_t capacity) {
    return json_result({{"schema", "phoenix-native-cuda-capabilities/1"},
                        {"host_frame_api_versions", {2}},
                        {"frame_contract", "reset-rgb/1"},
                        {"architectures", {"sm_120"}},
                        {"motion", false},
                        {"temporal_reuse", false},
                        {"advanced_controls", false},
                        {"model_weights_included", false}},
                       destination, capacity);
}
extern "C" int phoenix_runtime_frame(void*, const float*, const float*, const float*, const float*,
                                     const char*, float*) {
    return unsupported("Native CUDA requires bounded frame API V2");
}
extern "C" int phoenix_runtime_frame_v3(void*, const phoenix_frame_v3*) {
    return unsupported("Native CUDA preview does not implement V3");
}
extern "C" int phoenix_runtime_replace_model(void*, const char*) {
    return unsupported("Recreate the native CUDA session to replace the model");
}

extern "C" __attribute__((visibility("default"))) int
phoenix_linux_frame_rgba8(void* handle, const phoenix_rgba8_frame* frame) {
    try {
        if (!handle || !frame || frame->struct_bytes != sizeof(*frame) || frame->version != 1 ||
            frame->format > 1 || frame->flags != 1 || !frame->data)
            throw std::runtime_error("Invalid native RGBA8 frame header");
        auto& session = *static_cast<Session*>(handle);
        std::lock_guard guard(session.mutex);
        const auto width = session.executor.width, height = session.executor.height;
        const uint64_t row = uint64_t(width) * 4;
        const uint64_t stride = frame->row_stride_bytes ? frame->row_stride_bytes : row;
        if (frame->width != width || frame->height != height || stride < row ||
            stride > (UINT64_MAX - row) / height)
            throw std::runtime_error("Invalid native RGBA8 dimensions/stride");
        const uint64_t required = stride * (height - 1) + row;
        if (required > frame->capacity_bytes || required > SIZE_MAX ||
            required > UINTPTR_MAX - reinterpret_cast<uintptr_t>(frame->data))
            throw std::runtime_error("Insufficient native RGBA8 capacity");
        const auto* output = session.executor.run_rgba8(static_cast<const uint8_t*>(frame->data),
                                                        size_t(stride), frame->format == 1);
        auto* destination = static_cast<uint8_t*>(frame->data);
        for (unsigned y = 0; y < height; ++y)
            std::memcpy(destination + size_t(y) * size_t(stride), output + size_t(y) * width * 4,
                        size_t(width) * 4);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}
