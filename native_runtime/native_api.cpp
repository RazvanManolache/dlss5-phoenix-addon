#define PHOENIX_NATIVE_BUILD
#include "phoenix_native.h"
#include "../src/native/support/resolution_bridge.h"
#include "../src/native/support/dispatch.h"
#include <chrono>
#include <thread>

using Microsoft::WRL::ComPtr;
using Json = phoenix_bundle::Json;
using phoenix_native::hr;
using phoenix_bundle::require;
namespace spatial = phoenix_addon::phoenix::direct;
namespace {
thread_local std::string last_error;
template <class Function> int protect(Function function) noexcept {
    try {
        last_error.clear();
        function();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    } catch (...) {
        last_error = "Unknown native library error";
        return -1;
    }
}
std::string file_digest(const std::filesystem::path& path) {
    struct Mapping {
        HANDLE file = INVALID_HANDLE_VALUE, mapping = nullptr;
        const uint8_t* data = nullptr;
        ~Mapping() {
            if (data)
                UnmapViewOfFile(data);
            if (mapping)
                CloseHandle(mapping);
            if (file != INVALID_HANDLE_VALUE)
                CloseHandle(file);
        }
    } view;
    view.file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
    require(view.file != INVALID_HANDLE_VALUE, "Cannot open native bundle/model file");
    LARGE_INTEGER size{};
    require(GetFileSizeEx(view.file, &size) && size.QuadPart > 0 &&
                uint64_t(size.QuadPart) <= SIZE_MAX,
            "Invalid native bundle/model file size");
    view.mapping = CreateFileMappingW(view.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    require(view.mapping != nullptr, "Cannot map native bundle/model file");
    view.data = static_cast<const uint8_t*>(MapViewOfFile(view.mapping, FILE_MAP_READ, 0, 0, 0));
    require(view.data != nullptr, "Cannot read native bundle/model file");
    return phoenix_model::weight_digest(view.data, size_t(size.QuadPart));
}
bool same_device(ID3D12Device* device, ID3D12DeviceChild* child) {
    ComPtr<ID3D12Device> owner;
    return child && SUCCEEDED(child->GetDevice(IID_PPV_ARGS(&owner))) &&
           phoenix_native::device_owns(device, owner.Get());
}
struct Plane {
    ComPtr<ID3D12Resource> image, transfer;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    uint64_t bytes = 0;
};
void buffer_extent(const void* pointer, size_t capacity, size_t stride, unsigned width,
                   unsigned height, unsigned pixel_bytes) {
    const size_t row = size_t(width) * pixel_bytes;
    require(pointer && stride >= row && stride <= SIZE_MAX / height &&
                capacity >= stride * (height - 1) + row,
            "Host plane capacity/stride invalid");
}
}

struct PhoenixNativeSession {
    // Member order makes the networks/descriptors retire before their API/device.
    const std::thread::id thread = std::this_thread::get_id();
    unsigned width, height, output_width, output_height, neural_width, neural_height;
    phoenix_addon::phoenix::Resolution geometry;
    bool spatial_only, poisoned = false, history = false, in_flight = false;
    uint64_t submitted = 0, frequency = 0, completed_frames = 0;
    double gpu_ms = 0, call_ms = 0;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> command;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12QueryHeap> queries;
    ComPtr<ID3D12Resource> timestamps;
    ComPtr<ID3D12Resource> retained_color, retained_motion, retained_output;
    phoenix_native::memory::Ledger ledger;
    std::shared_ptr<phoenix_native::memory::Generation> accounting;
    ComPtr<ID3D12Resource> input_color, input_motion;
    Plane host_color, host_motion, host_output;
    std::unique_ptr<phoenix_addon::FrameDispatch> dispatch;
    std::unique_ptr<phoenix_native::Network> network;
    std::unique_ptr<spatial::ResolutionBridge> resize;

    void check_thread() const {
        require(thread == std::this_thread::get_id(),
                "Native session belongs to its creator thread");
    }
    PhoenixNativeSession(const Json& config, ID3D12Device* supplied_device,
                         ID3D12CommandQueue* supplied_queue)
        : width(config.at("width")), height(config.at("height")),
          output_width(config.value("output_width", width)),
          output_height(config.value("output_height", height)),
          neural_width(config.value("neural_width", width)),
          neural_height(config.value("neural_height", height)),
          spatial_only(config.value("spatial_only", false)) {
        require(width && height && neural_width && neural_height && output_width && output_height &&
                    width <= 16384 && height <= 16384 && output_width <= 16384 &&
                    output_height <= 16384 && neural_width <= 16384 && neural_height <= 16384,
                "Invalid native/source extent");
        require(bool(supplied_device) == bool(supplied_queue), "Supply both device and queue");
        if (supplied_device) {
            require(same_device(supplied_device, supplied_queue) &&
                        supplied_queue->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT,
                    "Device/queue ownership or DIRECT type mismatch");
            device = supplied_device;
            queue = supplied_queue;
        } else {
            ComPtr<IDXGIFactory6> factory;
            hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
            ComPtr<IDXGIAdapter1> adapter;
            if (config.value("warp", false)) {
                require(spatial_only, "WARP cannot execute native RTX kernels");
                hr(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
            } else if (config.contains("adapter_index")) {
                hr(factory->EnumAdapters1(config.at("adapter_index").get<unsigned>(), &adapter));
            } else {
                for (unsigned index = 0;; ++index) {
                    ComPtr<IDXGIAdapter1> candidate;
                    if (factory->EnumAdapterByGpuPreference(
                            index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                            IID_PPV_ARGS(&candidate)) == DXGI_ERROR_NOT_FOUND)
                        break;
                    DXGI_ADAPTER_DESC1 description{};
                    hr(candidate->GetDesc1(&description));
                    if (spatial_only || description.VendorId == 0x10de) {
                        adapter = candidate;
                        break;
                    }
                }
                require(adapter != nullptr, "No NVIDIA adapter available for native RTX");
            }
            hr(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
            D3D12_COMMAND_QUEUE_DESC description{};
            hr(device->CreateCommandQueue(&description, IID_PPV_ARGS(&queue)));
        }
        hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                          IID_PPV_ARGS(&allocator)));
        hr(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                     IID_PPV_ARGS(&command)));
        hr(command->Close());
        hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        hr(queue->GetTimestampFrequency(&frequency));
        require(frequency != 0, "GPU timestamp frequency unavailable");
        D3D12_QUERY_HEAP_DESC query_description{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 0};
        hr(device->CreateQueryHeap(&query_description, IID_PPV_ARGS(&queries)));
        timestamps = phoenix_native::buffer(device.Get(), 16, D3D12_HEAP_TYPE_READBACK,
                                            D3D12_RESOURCE_STATE_COPY_DEST);
        accounting = ledger.begin(neural_width, neural_height);
        input_color = image(neural_width, neural_height, DXGI_FORMAT_R16G16B16A16_FLOAT);
        input_motion = image(neural_width, neural_height, DXGI_FORMAT_R16G16_FLOAT);
        if (!spatial_only) {
            const auto folder =
                std::filesystem::u8path(config.at("bundle_path").get<std::string>());
            require(folder.is_absolute(), "Native bundle path must be absolute");
            phoenix_bundle::Bundle bundle(
                folder, file_digest,
                phoenix_model::WeightSource::config(config.at("weight_source")));
            auto weights = bundle.load_weights();
            auto kernels = phoenix_kernel::load_catalog(phoenix_native::catalog(bundle));
            const auto& plan =
                bundle.select_frame(neural_width, neural_height, [&](const Json& frame) {
                    phoenix_native::preflight(frame, kernels);
                });
            dispatch = std::make_unique<phoenix_addon::FrameDispatch>();
            const auto counts = config.value("repeats", std::array<int, 8>{1, 1, 1, 1, 1, 1, 1, 1});
            dispatch->configure_repeats(device.Get(), plan, counts);
            network = std::make_unique<phoenix_native::Network>(device.Get(), dispatch->api(), plan,
                                                                weights, kernels, input_color.Get(),
                                                                input_motion.Get(), accounting);
        }
        geometry =
            phoenix_addon::phoenix::fit_resolution(width, height, neural_width, neural_height);
        geometry.output_width = output_width;
        geometry.output_height = output_height;
        if (spatial_only || neural_width != width || neural_height != height ||
            output_width != width || output_height != height)
            resize = std::make_unique<spatial::ResolutionBridge>(
                device.Get(), geometry, DXGI_FORMAT_R16G16B16A16_FLOAT, accounting);
    }
    ComPtr<ID3D12Resource> image(unsigned w, unsigned h, DXGI_FORMAT format) {
        auto resource = phoenix_native::image(device.Get(), w, h, format,
                                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::memory::resource(device.Get(), resource.Get(), accounting,
                                         phoenix_native::memory::Kind::bridge);
        return resource;
    }
    void validate(ID3D12Resource* resource, DXGI_FORMAT format, bool output = false) {
        require(same_device(device.Get(), resource), "Texture belongs to a different device");
        const auto description = resource->GetDesc();
        require(description.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
                    description.Width == (output ? output_width : width) &&
                    description.Height == (output ? output_height : height) &&
                    description.DepthOrArraySize == 1 && description.MipLevels == 1 &&
                    description.SampleDesc.Count == 1 && description.Format == format &&
                    !(description.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE),
                "Texture extent/format mismatch");
    }
    void begin() {
        hr(allocator->Reset());
        hr(command->Reset(allocator.Get(), nullptr));
    }
    void wait(uint64_t value) {
        require(fence->GetCompletedValue() != UINT64_MAX, "Native GPU device removed");
        if (fence->GetCompletedValue() < value) {
            HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            require(event != nullptr, "Fence event allocation failed");
            const HRESULT registered = fence->SetEventOnCompletion(value, event);
            const DWORD result =
                SUCCEEDED(registered) ? WaitForSingleObject(event, 30000) : WAIT_FAILED;
            CloseHandle(event);
            hr(registered);
            require(result == WAIT_OBJECT_0, "Native GPU fence wait failed/timed out");
        }
        hr(device->GetDeviceRemovedReason());
        in_flight = false;
        retained_color.Reset();
        retained_motion.Reset();
        retained_output.Reset();
    }
    void submit() {
        hr(command->Close());
        ID3D12CommandList* lists[]{command.Get()};
        in_flight = true; // Retain all referenced objects even if Signal fails.
        queue->ExecuteCommandLists(1, lists);
        hr(queue->Signal(fence.Get(), ++submitted));
    }
    void drain() {
        if (!in_flight || FAILED(device->GetDeviceRemovedReason()))
            return;
        hr(queue->Signal(fence.Get(), ++submitted));
        wait(submitted);
    }
    void copy_input(ID3D12Resource* source, D3D12_RESOURCE_STATES state, ID3D12Resource* target,
                    bool motion) {
        if (resize) {
            resize->reduce(command.Get(), source, state, target, motion,
                           float(geometry.image_width) / width,
                           float(geometry.image_height) / height);
            return;
        }
        phoenix_native::transition(command.Get(), source, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        phoenix_native::transition(command.Get(), target,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_COPY_DEST);
        command->CopyResource(target, source);
        phoenix_native::transition(command.Get(), target, D3D12_RESOURCE_STATE_COPY_DEST,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::transition(command.Get(), source, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
    }
    void frame(ID3D12Resource* color, D3D12_RESOURCE_STATES color_state, ID3D12Resource* motion,
               D3D12_RESOURCE_STATES motion_state, ID3D12Resource* output,
               D3D12_RESOURCE_STATES output_state, const PhoenixNativeControls* options) {
        check_thread();
        require(!poisoned, "Native session failed; destroy and recreate it");
        validate(color, DXGI_FORMAT_R16G16B16A16_FLOAT);
        validate(output, DXGI_FORMAT_R16G16B16A16_FLOAT, true);
        if (motion)
            validate(motion, DXGI_FORMAT_R16G16_FLOAT);
        require(color != output && motion != output && color != motion, "Texture alias rejected");
        PhoenixNativeControls defaults;
        phoenix_native_default_controls(&defaults);
        if (!options)
            options = &defaults;
        require(options->struct_bytes == sizeof(*options),
                "Native control structure version mismatch");
        phoenix_addon::Controls controls;
        controls.reset = options->reset != 0;
        controls.motion_x = options->motion_scale_x;
        controls.motion_y = options->motion_scale_y;
        controls.tone = options->local_tone;
        controls.structure = options->local_structure;
        controls.skin = options->skin_structure;
        controls.auto_mask = options->auto_mask != 0;
        controls.inverted = options->inverted_depth != 0;
        // Reject bad controls before opening a command list or touching history.
        auto resolved = phoenix_addon::resolve_controls(
            controls, neural_width, neural_height, motion != nullptr, history,
            controls.reset || !network ? 0 : network->next_sampling_index());
        const auto started = std::chrono::steady_clock::now();
        retained_color = color;
        retained_motion = motion;
        retained_output = output;
        try {
            begin();
            if (resize)
                resize->begin_frame(submitted, submitted + 1);
            copy_input(color, color_state, input_color.Get(), false);
            if (motion)
                copy_input(motion, motion_state, input_motion.Get(), true);
            command->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            ID3D12Resource* result = input_color.Get();
            if (network) {
                dispatch->prepare_repeats(command.Get());
                phoenix_addon::FrameDispatch::Scope scope(*dispatch, command.Get(), resolved);
                result = network->record(command.Get(), !resolved.recurrent, 0, queue.Get());
                scope.finish();
            }
            phoenix_native::transition(command.Get(), result,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            if (resize)
                resize->present_source(command.Get(), result, color, color_state, output,
                                       output_state);
            else {
                phoenix_native::transition(command.Get(), output, output_state,
                                           D3D12_RESOURCE_STATE_COPY_DEST);
                command->CopyResource(output, result);
                phoenix_native::transition(command.Get(), output, D3D12_RESOURCE_STATE_COPY_DEST,
                                           output_state);
            }
            phoenix_native::transition(command.Get(), result, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            command->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
            command->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2,
                                      timestamps.Get(), 0);
            submit();
            if (network)
                network->commit(command.Get());
            wait(submitted);
            history = motion != nullptr;
            ++completed_frames;
            uint64_t* values = nullptr;
            D3D12_RANGE range{0, 16}, empty{};
            hr(timestamps->Map(0, &range, reinterpret_cast<void**>(&values)));
            gpu_ms = double(values[1] - values[0]) * 1000.0 / frequency;
            timestamps->Unmap(0, &empty);
            call_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                started)
                          .count();
        } catch (...) {
            poisoned = true;
            if (network)
                network->invalidate();
            throw;
        }
    }
    void prepare_plane(Plane& plane, DXGI_FORMAT format, bool readback) {
        if (plane.image)
            return;
        Plane prepared;
        prepared.image =
            image(readback ? output_width : width, readback ? output_height : height, format);
        const auto description = prepared.image->GetDesc();
        device->GetCopyableFootprints(&description, 0, 1, 0, &prepared.footprint, nullptr, nullptr,
                                      &prepared.bytes);
        prepared.transfer = phoenix_native::buffer(
            device.Get(), prepared.bytes,
            readback ? D3D12_HEAP_TYPE_READBACK : D3D12_HEAP_TYPE_UPLOAD,
            readback ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_GENERIC_READ);
        phoenix_native::memory::resource(device.Get(), prepared.transfer.Get(), accounting,
                                         phoenix_native::memory::Kind::upload);
        plane = std::move(prepared);
    }
    void upload(Plane& plane, const void* data, size_t stride, unsigned row_bytes) {
        void* mapped = nullptr;
        D3D12_RANGE empty{};
        hr(plane.transfer->Map(0, &empty, &mapped));
        for (unsigned row = 0; row < height; ++row)
            std::memcpy(static_cast<char*>(mapped) + plane.footprint.Offset +
                            size_t(row) * plane.footprint.Footprint.RowPitch,
                        static_cast<const char*>(data) + row * stride, row_bytes);
        plane.transfer->Unmap(0, nullptr);
        phoenix_native::transition(command.Get(), plane.image.Get(),
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION from{plane.transfer.Get(),
                                         D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, plane.footprint};
        D3D12_TEXTURE_COPY_LOCATION to{plane.image.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                                       0};
        command->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        phoenix_native::transition(command.Get(), plane.image.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
};

extern "C" {
void phoenix_native_default_controls(PhoenixNativeControls* controls) {
    if (controls)
        *controls = {sizeof(*controls), 0, 1, 1, 1, 1, -1, 0, 1};
}
PhoenixNativeSession* phoenix_native_create(const char* config, ID3D12Device* device,
                                            ID3D12CommandQueue* queue) {
    PhoenixNativeSession* result = nullptr;
    protect([&] {
        require(config != nullptr, "Missing native configuration");
        result = new PhoenixNativeSession(Json::parse(config), device, queue);
    });
    return result;
}
int phoenix_native_frame_d3d12(PhoenixNativeSession* session, ID3D12Resource* color,
                               D3D12_RESOURCE_STATES color_state, ID3D12Resource* motion,
                               D3D12_RESOURCE_STATES motion_state, ID3D12Resource* output,
                               D3D12_RESOURCE_STATES output_state,
                               const PhoenixNativeControls* controls) {
    return protect([&] {
        require(session != nullptr, "Null native session");
        session->frame(color, color_state, motion, motion_state, output, output_state, controls);
    });
}
int phoenix_native_frame_host(PhoenixNativeSession* session, const void* color, size_t color_bytes,
                              size_t color_stride, const void* motion, size_t motion_bytes,
                              size_t motion_stride, void* output, size_t output_bytes,
                              size_t output_stride, const PhoenixNativeControls* controls) {
    return protect([&] {
        require(session != nullptr, "Null native session");
        session->check_thread();
        require(!session->poisoned, "Native session failed; recreate it");
        buffer_extent(color, color_bytes, color_stride, session->width, session->height, 8);
        buffer_extent(output, output_bytes, output_stride, session->output_width,
                      session->output_height, 8);
        if (motion)
            buffer_extent(motion, motion_bytes, motion_stride, session->width, session->height, 4);
        if (controls)
            require(controls->struct_bytes == sizeof(*controls),
                    "Native control structure version mismatch");
        if (controls)
            require(std::isfinite(controls->motion_scale_x) &&
                        std::isfinite(controls->motion_scale_y) &&
                        std::isfinite(controls->local_tone) &&
                        std::isfinite(controls->local_structure) &&
                        std::isfinite(controls->skin_structure),
                    "Non-finite native control");
        session->prepare_plane(session->host_color, DXGI_FORMAT_R16G16B16A16_FLOAT, false);
        session->prepare_plane(session->host_output, DXGI_FORMAT_R16G16B16A16_FLOAT, true);
        if (motion)
            session->prepare_plane(session->host_motion, DXGI_FORMAT_R16G16_FLOAT, false);
        const auto started = std::chrono::steady_clock::now();
        try {
            session->begin();
            session->upload(session->host_color, color, color_stride, session->width * 8);
            if (motion)
                session->upload(session->host_motion, motion, motion_stride, session->width * 4);
            session->submit();
            session->wait(session->submitted);
            const auto state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            session->frame(session->host_color.image.Get(), state,
                           motion ? session->host_motion.image.Get() : nullptr, state,
                           session->host_output.image.Get(), state, controls);
            session->begin();
            auto& plane = session->host_output;
            phoenix_native::transition(session->command.Get(), plane.image.Get(), state,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION from{plane.image.Get(),
                                             D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0};
            D3D12_TEXTURE_COPY_LOCATION to{
                plane.transfer.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, plane.footprint};
            session->command->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            phoenix_native::transition(session->command.Get(), plane.image.Get(),
                                       D3D12_RESOURCE_STATE_COPY_SOURCE, state);
            session->submit();
            session->wait(session->submitted);
            void* mapped = nullptr;
            D3D12_RANGE range{0, size_t(plane.bytes)}, empty{};
            hr(plane.transfer->Map(0, &range, &mapped));
            for (unsigned row = 0; row < session->output_height; ++row)
                std::memcpy(static_cast<char*>(output) + row * output_stride,
                            static_cast<const char*>(mapped) + plane.footprint.Offset +
                                size_t(row) * plane.footprint.Footprint.RowPitch,
                            size_t(session->output_width) * 8);
            plane.transfer->Unmap(0, &empty);
            session->call_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();
        } catch (...) {
            session->poisoned = true;
            throw;
        }
    });
}
int phoenix_native_device(PhoenixNativeSession* session, ID3D12Device** device,
                          ID3D12CommandQueue** queue) {
    if (device)
        *device = nullptr;
    if (queue)
        *queue = nullptr;
    return protect([&] {
        require(session != nullptr, "Null native session");
        session->check_thread();
        if (device) {
            *device = session->device.Get();
            (*device)->AddRef();
        }
        if (queue) {
            *queue = session->queue.Get();
            (*queue)->AddRef();
        }
    });
}
int phoenix_native_destroy(PhoenixNativeSession* session) {
    return protect([&] {
        if (session) {
            session->check_thread();
            session->drain();
            delete session;
        }
    });
}
size_t phoenix_native_status(PhoenixNativeSession* session, char* destination, size_t capacity) {
    size_t result = 0;
    protect([&] {
        require(session != nullptr, "Null native session");
        session->check_thread();
        const auto text =
            Json{{"backend", session->spatial_only ? "spatial_d3d12" : "native_rtx"},
                 {"width", session->width},
                 {"height", session->height},
                 {"output_width", session->output_width},
                 {"output_height", session->output_height},
                 {"neural_width", session->neural_width},
                 {"neural_height", session->neural_height},
                 {"gpu_processing_ms", session->gpu_ms},
                 {"complete_call_ms", session->call_ms},
                 {"gpu_timing_scope",
                  "model plus output resize/copy; excludes input preparation and host transfers"},
                 {"frames", session->completed_frames},
                 {"failed", session->poisoned},
                 {"owned_memory", session->ledger.snapshot()}}
                .dump();
        result = text.size() + 1;
        if (destination && capacity >= result)
            std::memcpy(destination, text.c_str(), result);
    });
    return result;
}
const char* phoenix_native_error() {
    return last_error.c_str();
}
}
