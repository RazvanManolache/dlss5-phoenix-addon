#define PHOENIX_NATIVE_BUILD
#include "phoenix_native.h"
#include "../src/native/session/session.h"
#include "../src/native/support/resolution_bridge.h"
#include "../src/native/support/dispatch.h"
#include "../src/native/support/appearance_gpu.h"
#include "frame.h"
#include "advanced_frame.h"
#include "broker_client.h"
#include "../src/pipeline/d3d12_motion.h"
#include "../src/pipeline/json.h"
#include "../src/pipeline/nis.h"
#include "../src/pipeline/tiny.h"
#include "../src/pipeline/recipe.h"
#include "../src/pipeline/image_ops.h"
#include <chrono>
#include <thread>
#include <d3d12sdklayers.h>

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
    return phoenix_model::mapped_file_digest(path);
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
    struct Graph;
    std::shared_ptr<Graph> graph;
    Json configuration;
    phoenix_pipeline::Tuning tuning;
    phoenix_pipeline::Tuning requested_tuning;
    phoenix_pipeline::Options requested_pipeline;
    phoenix_pipeline::LatencyPolicy latency_policy;
    bool rebuild_graph=false;
    unsigned transformer_blocks=8;
    uint64_t graph_evaluations=0;
    Json pipeline_telemetry=Json::object();
    Json completed_tuning=nullptr,completed_pipeline=nullptr;
    bool graph_owner=false;
    Json portable_stage_reports()const;
    bool neural_budget_enabled()const{return !spatial_only&&!portable_leaf&&!configuration.value("offline",false)&&(tuning.target_fps>0||tuning.max_time_ms>0);}
    void update_neural_budget();
    void initialize_graph();
    void frame_graph(ID3D12Resource*,D3D12_RESOURCE_STATES,ID3D12Resource*,D3D12_RESOURCE_STATES,
        ID3D12Resource*,D3D12_RESOURCE_STATES,const PhoenixNativeControls*,const PhoenixNativeFrameV2*);
    // Member order makes the networks/descriptors retire before their API/device.
    const std::thread::id thread = std::this_thread::get_id();
    unsigned width, height, output_width, output_height, neural_width, neural_height;
    phoenix_addon::phoenix::Resolution geometry;
    bool spatial_only, portable=false, portable_leaf=false, poisoned = false, history = false, in_flight = false;
    std::unique_ptr<phoenix_cpp::NativeSession> portable_session;
    Json portable_status=Json::object();
    Plane portable_color_readback,portable_motion_readback,portable_output_upload;
    void frame_portable(ID3D12Resource*,D3D12_RESOURCE_STATES,ID3D12Resource*,D3D12_RESOURCE_STATES,
        ID3D12Resource*,D3D12_RESOURCE_STATES,const PhoenixNativeControls*,const PhoenixNativeFrameV2*);
    uint64_t submitted = 0, frequency = 0, completed_frames = 0;
    double gpu_ms = 0, call_ms = 0;
    double pending_source_age_ms=0,previous_presentation_tail_ms=0,completion_age_ms=0;
    bool caller_latency_pending=false,completed_caller_latency=false;
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
    Plane host_mask, host_ui, host_alpha, host_backbuffer, host_distortion;
    ComPtr<ID3D12Resource> input_mask, original_color, original_motion, previous_field, display_delta, display_output, unwarped;
    std::unique_ptr<spatial::Pass> grade, compose, warp_color, warp_motion, warp_mask, warp_output;
    std::vector<ComPtr<ID3D12Resource>> retained_optional;
    bool had_mask=false, had_distortion=false;
    bool last_advanced=false,last_ui=false;
    unsigned last_style=0;float last_intensity=1;
    std::string packed_weights_sha256;
    phoenix_model::WeightSource weight_source;
    std::unique_ptr<phoenix_addon::FrameDispatch> dispatch;
    std::unique_ptr<phoenix_native::Network> network;
    std::unique_ptr<spatial::ResolutionBridge> resize;
    phoenix_pipeline::Options pipeline;
    std::unique_ptr<phoenix_pipeline::D3D12Motion> motion_provider;
    bool engine_motion_valid = true, measure_gpu = true;
    unsigned gpu_sample_interval = 1;
    bool gpu_sampled_last_frame = false;
    std::string motion_status = "Engine motion or zero fallback";
    PhoenixNativeFrameInfo next_frame_info{};
    bool frame_info_pending = false;
    phoenix_pipeline::Discontinuity source_discontinuity;
    std::unique_ptr<phoenix_pipeline::NisScaler> nis;
    std::unique_ptr<spatial::Pass> scale_alpha;
    ComPtr<ID3D12Resource> scaled_image, scaled_display, motion_debug;
    std::string scaler_status = "Existing spatial scaler";
    std::unique_ptr<phoenix_pipeline::TinyUpscaler> tiny;
    std::filesystem::path tiny_model_path;

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
        configuration=config;
        tuning=phoenix_pipeline::Tuning::read(config.value("tuning",Json::object()));
        requested_tuning=tuning;
        transformer_blocks=config.contains("tuning")?tuning.blocks:config.value("transformer_blocks",8u);
        require(transformer_blocks>=1&&transformer_blocks<=8,"Invalid central block count");
        if (config.contains("pipeline")) {
            pipeline = phoenix_pipeline::options_from_json(config.at("pipeline"));
            measure_gpu = pipeline.monitoring == int(phoenix_pipeline::Monitoring::Detailed);
            gpu_sample_interval = unsigned(pipeline.sample_interval);
        }
        engine_motion_valid = config.value("engine_motion_valid", true);
        requested_pipeline=pipeline;
        require(!pipeline.chain_enabled || pipeline.scaler != 2 ||
            !pipeline.enabled[unsigned(phoenix_pipeline::Stage::Scale)], "RTX VSR is deferred; choose an available scaling provider");
        tiny_model_path = std::filesystem::u8path(config.value("tiny_model_path", std::string()));
        portable=!spatial_only&&config.contains("portable_runtime");
        portable_leaf=portable&&config.value("_portable_leaf",false);
        graph_owner=!portable_leaf&&(portable||pipeline.priority!=0||!phoenix_pipeline::direct_tuning(tuning)||
            !phoenix_pipeline::default_chain(pipeline,width,height,output_width,output_height));
        require(!portable_leaf||(width==output_width&&height==output_height&&width==neural_width&&height==neural_height&&
            !pipeline.chain_enabled&&!pipeline.motion_enabled&&!pipeline.scene_detection),"Invalid portable neural stage contract");
        if (!spatial_only) {
            weight_source = phoenix_model::WeightSource::config(config.at("weight_source"));
            phoenix_model::require_implemented(weight_source.model());
            phoenix_model::native_layout(weight_source.model().architecture);
            if (weight_source.model().architecture == phoenix_model::models::small_fp8.architecture)
                require(weight_source.external(), "Native RTX requires an external model source");
        }
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
            if (config.value("debug", false)) {
                ComPtr<ID3D12Debug> debug;
                hr(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
                debug->EnableDebugLayer();
            }
            ComPtr<IDXGIFactory6> factory;
            hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
            ComPtr<IDXGIAdapter1> adapter;
            if (config.value("warp", false)) {
                require(spatial_only||portable, "WARP cannot execute native RTX kernels");
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
                    if (spatial_only || portable || description.VendorId == 0x10de) {
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
        if (measure_gpu) hr(queue->GetTimestampFrequency(&frequency));
        require(!measure_gpu || frequency != 0, "GPU timestamp frequency unavailable");
        D3D12_QUERY_HEAP_DESC query_description{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 0};
        if (measure_gpu) {
            hr(device->CreateQueryHeap(&query_description, IID_PPV_ARGS(&queries)));
            timestamps = phoenix_native::buffer(device.Get(), 16, D3D12_HEAP_TYPE_READBACK,
                                                D3D12_RESOURCE_STATE_COPY_DEST);
        }
        accounting = ledger.begin(neural_width, neural_height);
        if(!graph_owner){
            input_color = image(neural_width, neural_height, DXGI_FORMAT_R16G16B16A16_FLOAT);
            input_motion = image(neural_width, neural_height, DXGI_FORMAT_R16G16_FLOAT);
        }
        if(portable_leaf){
            auto runner=config.at("portable_runtime");
            const auto root=std::filesystem::u8path(runner.at("runtime_root").get<std::string>());
            require(root.is_absolute(),"Portable runtime root must be absolute");
            const auto dll=root/"phoenix_runtime_cpp.dll";
            require(runner.contains("runtime_sha256")&&file_digest(dll)==runner.at("runtime_sha256").get<std::string>(),
                "Portable runtime checksum differs");
            if(runner.contains("files")){
                const auto& files=runner.at("files");require(files.is_object()&&files.size()<=512,"Invalid portable package manifest");
                for(const auto& [name,digest]:files.items()){
                    require(!name.empty()&&name.find(':')==std::string::npos&&name.find('\\')==std::string::npos,
                        "Invalid portable dependency path");
                    const auto relative=std::filesystem::u8path(name);require(!relative.is_absolute(),"Absolute portable dependency path");
                    for(const auto& part:relative)require(part!=".."&&part!=".","Portable dependency traversal rejected");
                    auto path=root/relative;
                    for(auto current=path;!current.empty();){const auto attributes=GetFileAttributesW(current.c_str());
                        require(attributes==INVALID_FILE_ATTRIBUTES||!(attributes&FILE_ATTRIBUTE_REPARSE_POINT),"Portable dependencies cannot traverse links");
                        const auto parent=current.parent_path();if(parent==current)break;current=parent;}
                    require(file_digest(path)==digest.get<std::string>(),"Portable dependency checksum differs");
                }
            }
            runner["weight_source"]=config.at("weight_source");
            // Separate native workers let two replacement renderers and Vulkan
            // branches coexist without violating the runner's public-session lock.
            runner["execution"]="broker";
            portable_session=std::make_unique<phoenix_cpp::NativeSession>();
            portable_session->open(root,runner,width,height);
            portable_status=Json::parse(portable_session->status_json());
        }
        if (!spatial_only&&!graph_owner&&!portable_leaf) {
            const auto folder =
                std::filesystem::u8path(config.at("bundle_path").get<std::string>());
            require(folder.is_absolute(), "Native bundle path must be absolute");
            phoenix_bundle::Bundle bundle(folder, file_digest, weight_source, true);
            auto weights = bundle.load_weight_view();
            packed_weights_sha256=bundle.weights.at("sha256").get<std::string>();
            auto kernels = phoenix_kernel::load_catalog(phoenix_native::catalog(bundle));
            const auto& plan =
                bundle.select_frame(neural_width, neural_height, [&](const Json& frame) {
                    phoenix_native::preflight(frame, kernels);
                });
            dispatch = std::make_unique<phoenix_addon::FrameDispatch>();
            const auto counts = config.contains("tuning")?tuning.repeats:config.value("repeats", std::array<int, 8>{1, 1, 1, 1, 1, 1, 1, 1});
            dispatch->configure_repeats(device.Get(), plan, counts);
            network = std::make_unique<phoenix_native::Network>(device.Get(), dispatch->api(), plan,
                                                                weights, kernels, input_color.Get(),
                                                                input_motion.Get(), accounting);
        }
        geometry =
            phoenix_addon::phoenix::fit_resolution(width, height, neural_width, neural_height);
        geometry.output_width = output_width;
        geometry.output_height = output_height;
        if (!graph_owner&&(spatial_only || neural_width != width || neural_height != height ||
            output_width != width || output_height != height))
            resize = std::make_unique<spatial::ResolutionBridge>(
                device.Get(), geometry, DXGI_FORMAT_R16G16B16A16_FLOAT, accounting);
        if(graph_owner)initialize_graph();
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
        const HRESULT closed = command->Close();
        if (FAILED(closed)) {
            std::string messages;
            ComPtr<ID3D12InfoQueue> info;
            if (SUCCEEDED(device.As(&info))) {
                const auto count = info->GetNumStoredMessages();
                for (UINT64 i = count > 16 ? count - 16 : 0; i < count; ++i) {
                    SIZE_T bytes = 0;
                    if (FAILED(info->GetMessage(i, nullptr, &bytes))) continue;
                    std::vector<uint8_t> storage(bytes);
                    auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
                    if (SUCCEEDED(info->GetMessage(i, message, &bytes))) {
                        messages += "\nD3D12 message " + std::to_string(message->ID) + ": ";
                        messages.append(message->pDescription,
                            message->DescriptionByteLength && message->pDescription[message->DescriptionByteLength-1]=='\0'
                                ? message->DescriptionByteLength-1 : message->DescriptionByteLength);
                    }
                }
            }
            try { hr(closed); }
            catch (const std::exception& error) { throw std::runtime_error(std::string(error.what()) + messages); }
        }
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
               D3D12_RESOURCE_STATES output_state, const PhoenixNativeControls* options,
               const PhoenixNativeFrameV2* extended=nullptr) {
        check_thread();
        require(!poisoned, "Native session failed; destroy and recreate it");
        if(portable_leaf){frame_portable(color,color_state,motion,motion_state,output,output_state,options,extended);return;}
        if(graph_owner){
            frame_graph(color,color_state,motion,motion_state,output,output_state,options,extended);return;
        }
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
        if(configuration.contains("tuning")){
            defaults=*options;
            defaults.local_tone=tuning.tone;defaults.local_structure=tuning.structure;
            defaults.skin_structure=tuning.skin;defaults.auto_mask=tuning.auto_mask;
            options=&defaults;
        }
        const bool mapped=extended && extended->control_mask.resource,
                   distorted=extended && extended->distortion.resource;
        if (extended) {
            require(!resize && !spatial_only, "Advanced native frame requires equal source/network/output extents");
            require(std::isfinite(extended->intensity)&&extended->intensity>=0&&extended->intensity<=2&&extended->style<=2,
                    "Invalid native appearance controls");
            require(!(extended->ui.resource&&extended->ui_alpha.resource), "Supply UI or UIAlpha, not both");
            const PhoenixNativeTextureV2* optional[]={&extended->control_mask,&extended->ui,&extended->ui_alpha,&extended->backbuffer,&extended->distortion};
            const DXGI_FORMAT formats[]={DXGI_FORMAT_R32G32B32A32_FLOAT,DXGI_FORMAT_R32G32B32A32_FLOAT,DXGI_FORMAT_R32_FLOAT,DXGI_FORMAT_R16G16B16A16_FLOAT,DXGI_FORMAT_R32G32B32A32_FLOAT};
            for(unsigned i=0;i<5;++i) {
                require(!optional[i]->reserved,"Reserved native texture fields");
                if(optional[i]->resource) {
                    validate(optional[i]->resource,formats[i]);
                    require(optional[i]->resource!=output&&optional[i]->resource!=color&&optional[i]->resource!=motion,"Optional texture aliases frame buffers");
                    for(unsigned j=0;j<i;++j)require(optional[i]->resource!=optional[j]->resource,"Optional textures alias");
                }
            }
            if(mapped&&!input_mask) {
                input_mask=image(width,height,DXGI_FORMAT_R32G32B32A32_FLOAT);
                network->bind_control_mask(input_mask.Get());
            }
            if(!grade) {
                grade=std::make_unique<spatial::Pass>(device.Get(),spatial::appearance_delta_shader,accounting);
                compose=std::make_unique<spatial::Pass>(device.Get(),spatial::appearance_compose_shader,accounting);
                warp_color=std::make_unique<spatial::Pass>(device.Get(),spatial::appearance_warp_shader,accounting);
                warp_motion=std::make_unique<spatial::Pass>(*warp_color,accounting);
                warp_mask=std::make_unique<spatial::Pass>(*warp_color,accounting);
                warp_output=std::make_unique<spatial::Pass>(*warp_color,accounting);
                original_color=image(width,height,DXGI_FORMAT_R16G16B16A16_FLOAT);
                original_motion=image(width,height,DXGI_FORMAT_R16G16_FLOAT);
                previous_field=image(width,height,DXGI_FORMAT_R32G32B32A32_FLOAT);
                display_delta=image(width,height,DXGI_FORMAT_R32G32B32A32_FLOAT);
                display_output=image(width,height,DXGI_FORMAT_R16G16B16A16_FLOAT);
                unwarped=image(width,height,DXGI_FORMAT_R16G16B16A16_FLOAT);
            }
        }
        phoenix_addon::Controls controls;
        controls.reset = options->reset != 0;
        if(had_mask!=mapped || had_distortion!=distorted) controls.reset=1;
        controls.motion_x = options->motion_scale_x;
        controls.motion_y = options->motion_scale_y;
        controls.tone = options->local_tone;
        controls.structure = options->local_structure;
        controls.skin = options->skin_structure;
        controls.auto_mask = options->auto_mask != 0;
        controls.inverted = options->inverted_depth != 0;
        if (frame_info_pending) {
            const auto reason = source_discontinuity.observe({next_frame_info.stream, next_frame_info.sequence,
                next_frame_info.timestamp_ns, width, height}, controls.reset, false);
            controls.reset = controls.reset || reason != phoenix_pipeline::ResetReason::None;
        }
        if (pipeline.motion_enabled || pipeline.scene_detection) {
            require(!extended || !extended->distortion.resource,
                    "Estimated motion and bidirectional distortion require separate stages");
            if (!motion_provider)
                motion_provider = std::make_unique<phoenix_pipeline::D3D12Motion>(device.Get(), queue.Get(), fence.Get());
            retained_color = color;
            try {
                const auto estimated = motion_provider->prepare(color, color_state,
                    (frame_info_pending ? next_frame_info.engine_motion_valid != 0 : engine_motion_valid) ? motion : nullptr,
                    pipeline, controls.reset,
                    frame_info_pending ? next_frame_info.timestamp_ns : int64_t(completed_frames) * 16666667,
                    frame_info_pending ? next_frame_info.stream : 0,
                    frame_info_pending ? std::optional<uint64_t>(next_frame_info.sequence) : std::nullopt);
                motion_status = motion_provider->status();
                motion = estimated.motion;
                motion_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                controls.reset = controls.reset || estimated.reset;
            } catch (...) {
                // A failed wait may leave provider work referencing the source.
                // Retain the session until its queue has actually retired.
                poisoned = true; in_flight = true; throw;
            }
        }
        // Reject bad controls before opening a command list or touching history.
        auto resolved = phoenix_addon::resolve_controls(
            controls, neural_width, neural_height, motion != nullptr, history,
            controls.reset || !network ? 0 : network->next_sampling_index());
        if(extended) {
            resolved.control_mask_handle=mapped?network->control_mask_handle:0;
            resolved.style=float(extended->style)/128.f;
        }
        const bool measure_call = measure_gpu || pipeline.monitoring != 0 || neural_budget_enabled();
        const bool sample_gpu = measure_gpu && completed_frames % gpu_sample_interval == 0;
        const auto started = measure_call ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        retained_color = color;
        retained_motion = motion;
        retained_output = output;
        try {
            begin();
            if (resize)
                resize->begin_frame(submitted, submitted + 1);
            const auto shader_state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            auto record_warp=[&](spatial::Pass& pass,ID3D12Resource* source,ID3D12Resource* field,
                                ID3D12Resource* prior,ID3D12Resource* destination,unsigned mode) {
                spatial::Constants c{};c.width=width;c.height=height;c.mode=mode;
                c.divider=options->motion_scale_x;c.gain=options->motion_scale_y;
                phoenix_native::transition(command.Get(),destination,shader_state,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                pass.record(command.Get(),{source,field,prior},destination,c);
                phoenix_native::transition(command.Get(),destination,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,shader_state);
            };
            if(extended) {
                const PhoenixNativeTextureV2* optional[]={&extended->control_mask,&extended->ui,&extended->ui_alpha,&extended->backbuffer,&extended->distortion};
                retained_optional.clear();
                for(auto* plane:optional)if(plane->resource) {
                    retained_optional.emplace_back(plane->resource);
                    phoenix_native::transition(command.Get(),plane->resource,plane->state,shader_state);
                }
            }
            copy_input(color, color_state, extended?original_color.Get():input_color.Get(), false);
            if(distorted) record_warp(*warp_color,original_color.Get(),extended->distortion.resource,nullptr,input_color.Get(),0);
            else if(extended) copy_input(original_color.Get(),shader_state,input_color.Get(),false);
            if (motion)
                copy_input(motion, motion_state, input_motion.Get(), true);
            if(distorted&&motion) {
                // A separate source prevents read/write aliasing during reprojection.
                copy_input(motion,motion_state,original_motion.Get(),true);
                record_warp(*warp_motion,original_motion.Get(),extended->distortion.resource,
                    had_distortion&&!controls.reset?previous_field.Get():extended->distortion.resource,input_motion.Get(),2);
                resolved.motion_x=1.f/width;resolved.motion_y=1.f/height;
            }
            if(mapped) {
                if(distorted) record_warp(*warp_mask,extended->control_mask.resource,extended->distortion.resource,nullptr,input_mask.Get(),0);
                else copy_input(extended->control_mask.resource,shader_state,input_mask.Get(),false);
            }
            if (sample_gpu) command->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            ID3D12Resource* result = input_color.Get();
            if (network) {
                dispatch->prepare_repeats(command.Get());
                phoenix_addon::FrameDispatch::Scope scope(*dispatch, command.Get(), resolved,transformer_blocks);
                result = network->record(command.Get(), !resolved.recurrent, 0, queue.Get());
                scope.finish();
            }
            if(extended) {
                if(distorted) {record_warp(*warp_output,result,extended->distortion.resource,nullptr,unwarped.Get(),1);result=unwarped.Get();}
                const bool post=extended->intensity<1||mapped||extended->ui.resource||extended->ui_alpha.resource||extended->backbuffer.resource||
                    (extended->style&&extended->controls.local_tone>0);
                if(post) {
                    spatial::Constants c{};c.width=width;c.height=height;c.reset=extended->style;
                    c.mode=mapped?1u:0u;std::memcpy(&c.source_a,&controls.tone,sizeof(float));c.divider=extended->intensity;
                    phoenix_native::transition(command.Get(),display_delta.Get(),shader_state,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    grade->record(command.Get(),{original_color.Get(),result,extended->control_mask.resource},display_delta.Get(),c);
                    phoenix_native::transition(command.Get(),display_delta.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,shader_state);
                    auto* ui=extended->ui.resource?extended->ui.resource:extended->ui_alpha.resource;
                    c.mode=(ui?1u:0u)|(extended->ui_alpha.resource?2u:0u)|
                        (extended->ui.resource&&!extended->backbuffer.resource?4u:0u);
                    phoenix_native::transition(command.Get(),display_output.Get(),shader_state,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    compose->record(command.Get(),{display_delta.Get(),extended->backbuffer.resource?extended->backbuffer.resource:original_color.Get(),ui},display_output.Get(),c);
                    phoenix_native::transition(command.Get(),display_output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,shader_state);
                    result=display_output.Get();
                }
                if(distorted) {
                    phoenix_native::transition(command.Get(),extended->distortion.resource,shader_state,D3D12_RESOURCE_STATE_COPY_SOURCE);
                    phoenix_native::transition(command.Get(),previous_field.Get(),shader_state,D3D12_RESOURCE_STATE_COPY_DEST);
                    command->CopyResource(previous_field.Get(),extended->distortion.resource);
                    phoenix_native::transition(command.Get(),previous_field.Get(),D3D12_RESOURCE_STATE_COPY_DEST,shader_state);
                    phoenix_native::transition(command.Get(),extended->distortion.resource,D3D12_RESOURCE_STATE_COPY_SOURCE,shader_state);
                }
                const PhoenixNativeTextureV2* optional[]={&extended->control_mask,&extended->ui,&extended->ui_alpha,&extended->backbuffer,&extended->distortion};
                for(auto* plane:optional)if(plane->resource)phoenix_native::transition(command.Get(),plane->resource,shader_state,plane->state);
            }
            if (pipeline.motion_view && motion_provider) {
                if (!motion_debug) motion_debug = image(neural_width, neural_height, DXGI_FORMAT_R16G16B16A16_FLOAT);
                phoenix_native::transition(command.Get(), motion_debug.Get(), shader_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                if (motion_provider->visualize(command.Get(), motion_debug.Get(), unsigned(pipeline.motion_view))) result = motion_debug.Get();
                phoenix_native::transition(command.Get(), motion_debug.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, shader_state);
            }
            bool additional_scaling = false;
            if (pipeline.chain_enabled && pipeline.enabled[unsigned(phoenix_pipeline::Stage::Scale)] && pipeline.scaler == 3) {
                require(geometry.left == 0 && geometry.top == 0 && output_width == neural_width * 2 &&
                    output_height == neural_height * 2, "Tiny native upscaler requires a full-image 2x output");
                if (!tiny) tiny = std::make_unique<phoenix_pipeline::TinyUpscaler>(device.Get(),neural_width,neural_height,tiny_model_path);
                tiny->record(command.Get(),result,shader_state,output,output_state);
                scaler_status="Tiny native 2x (experimental)"; additional_scaling=true;
            }
            if (pipeline.chain_enabled && pipeline.enabled[unsigned(phoenix_pipeline::Stage::Scale)] && pipeline.scaler == 1) {
                if (!nis) {
                    nis = std::make_unique<phoenix_pipeline::NisScaler>(device.Get());
                    scaled_image = image(output_width, output_height, DXGI_FORMAT_R16G16B16A16_FLOAT);
                    scaled_display = image(output_width, output_height, DXGI_FORMAT_R16G16B16A16_FLOAT);
                    scale_alpha = std::make_unique<spatial::Pass>(device.Get(), spatial::resolution_shader, accounting);
                }
                additional_scaling = nis->record(command.Get(), result, shader_state, scaled_image.Get(), shader_state,
                    pipeline.sharpness, geometry.left, geometry.top, geometry.image_width, geometry.image_height);
                scaler_status = additional_scaling ? "NVIDIA Image Scaling" : "NIS requires 1x to 2x scaling per axis; existing scaler used";
                if (additional_scaling) {
                    spatial::Constants c{}; c.width = output_width; c.height = output_height; c.resolve = 3;
                    phoenix_native::transition(command.Get(), color, color_state, shader_state);
                    phoenix_native::transition(command.Get(), scaled_display.Get(), shader_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    scale_alpha->record(command.Get(), {scaled_image.Get(), color, nullptr}, scaled_display.Get(), c);
                    phoenix_native::transition(command.Get(), color, shader_state, color_state);
                    phoenix_native::transition(command.Get(), scaled_display.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    phoenix_native::transition(command.Get(), output, output_state, D3D12_RESOURCE_STATE_COPY_DEST);
                    command->CopyResource(output, scaled_display.Get());
                    phoenix_native::transition(command.Get(), output, D3D12_RESOURCE_STATE_COPY_DEST, output_state);
                    phoenix_native::transition(command.Get(), scaled_display.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, shader_state);
                }
            }
            phoenix_native::transition(command.Get(), result,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            if (additional_scaling) { /* Output and alpha were composed above. */ }
            else if (resize)
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
            if (sample_gpu) {
                command->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
                command->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2,
                                          timestamps.Get(), 0);
            }
            submit();
            if (motion_provider) motion_provider->consumed(submitted);
            if (network)
                network->commit(command.Get());
            wait(submitted);
            history = motion != nullptr;
            had_mask=mapped;had_distortion=distorted;retained_optional.clear();
            last_advanced=extended;last_ui=extended&&(extended->ui.resource||extended->ui_alpha.resource||extended->backbuffer.resource);
            last_style=extended?extended->style:0;last_intensity=extended?extended->intensity:1.f;
            auto applied=tuning;applied.blocks=transformer_blocks;applied.intensity=last_intensity;applied.style=last_style;
            applied.tone=options->local_tone;applied.structure=options->local_structure;applied.skin=options->skin_structure;applied.auto_mask=options->auto_mask!=0;
            if(!configuration.contains("tuning"))applied.repeats=configuration.value("repeats",applied.repeats);
            completed_tuning=applied.json();completed_pipeline=phoenix_pipeline::options_json(pipeline);
            ++completed_frames;
            frame_info_pending = false;
            gpu_sampled_last_frame = sample_gpu;
            if (sample_gpu) {
            uint64_t* values = nullptr;
            D3D12_RANGE range{0, 16}, empty{};
            hr(timestamps->Map(0, &range, reinterpret_cast<void**>(&values)));
            gpu_ms = double(values[1] - values[0]) * 1000.0 / frequency;
            timestamps->Unmap(0, &empty);
            }
            call_ms = measure_call ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                started)
                          .count() : 0;
            completion_age_ms=measure_call?pending_source_age_ms+call_ms:0;
            completed_caller_latency=measure_call&&caller_latency_pending;
            pending_source_age_ms=0;caller_latency_pending=false;
            update_neural_budget();
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

#include "native_pipeline.inc"
#include "native_portable.inc"

extern "C" {
unsigned phoenix_native_portable_contract(){return 1;}
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
int phoenix_native_frame_d3d12_v2(PhoenixNativeSession* session, const PhoenixNativeFrameV2* frame) {
    return protect([&] {
        require(session&&frame&&frame->struct_bytes==sizeof(*frame)&&frame->version==2,
                "Native advanced frame version mismatch");
        require(!frame->color.reserved&&!frame->motion.reserved&&!frame->output.reserved,
                "Reserved native frame fields");
        session->frame(frame->color.resource,frame->color.state,frame->motion.resource,frame->motion.state,
            frame->output.resource,frame->output.state,&frame->controls,frame);
    });
}
int phoenix_native_frame_host_v2(PhoenixNativeSession* session, const phoenix_frame_v3* request) {
    return protect([&] {
        using namespace phoenix_cpp;
        require(session&&request&&request->base.struct_bytes==sizeof(*request)&&
            request->base.version==PHOENIX_FRAME_VERSION_3&&!request->base.reserved,"Native bounded frame version mismatch");
        session->check_thread();require(!session->poisoned,"Native session failed; recreate it");
        require(!session->resize&&!session->spatial_only,"Advanced native host path requires equal neural/source/output extents");
        const unsigned w=session->width,h=session->height;const size_t pixels=size_t(w)*h;
        const auto& base=request->base;
        const auto output_stride=plane_stride(base.output,w,h,3);
        require(base.metadata_json,"Missing native host metadata");
        const auto metadata=Json::parse(base.metadata_json);
        require(metadata.is_object(),"Native host metadata must be an object");
        const auto controls=phoenix_cpp::Controls::from(metadata.value("controls",Json::object()));
        require(metadata.value("passes",1u)==1,"Native advanced adapter supports one network pass");
        auto color=copy_plane(base.color,w,h,3,true),motion=copy_plane(base.motion,w,h,2,false),
            depth=copy_plane(base.depth,w,h,1,false),validity=copy_plane(base.validity,w,h,1,false),
            mask=copy_plane(base.control_mask,w,h,4,false),ui=copy_plane(request->ui,w,h,4,false),
            alpha=copy_plane(request->ui_alpha,w,h,1,false),bb=copy_plane(request->backbuffer,w,h,3,false),
            field=copy_plane(request->distortion,w,h,4,false);
        require(depth.empty()&&validity.empty(),"Native standalone adapter has no depth/validity plane contract");
        require(ui.empty()||alpha.empty(),"Supply UI or UIAlpha, not both");
        require((mask.empty()&&base.control_profile==PHOENIX_CONTROL_NONE)||
            (!mask.empty()&&base.control_profile==PHOENIX_CONTROL_EXPERIMENTAL_RGB_1),"Native mask profile required");
        validate_control_mask(mask.empty()?nullptr:mask.data(),pixels);
        for(auto* plane:{&color,&ui,&alpha,&bb})validate_unit_plane(plane->empty()?nullptr:plane->data(),plane->size(),"Native proxy/alpha outside finite 0..1");
        for(float v:motion)require(std::isfinite(v),"Nonfinite native motion");
        for(float v:field)require(std::isfinite(v)&&std::fabs(v)<=1,"Invalid normalized native distortion");
        for(size_t p=0;!ui.empty()&&p<pixels;++p)for(unsigned c=0;c<3;++c)
            require(ui[p*4+c]<=ui[p*4+3],"Native UI must be premultiplied");
        auto declaration=[&](const char* name,bool present){require(!metadata.contains(name)||metadata.at(name).get<bool>()==present,"Native plane declaration conflict");};
        declaration("has_motion",!motion.empty());declaration("has_depth",false);declaration("has_history_validity",false);declaration("has_control_mask",!mask.empty());
        phx::Kernel<HostMath> math{};
        auto half_rgba=[&](const std::vector<float>& rgb) {
            std::vector<uint16_t> out(rgb.empty()?0:pixels*4);
            for(size_t p=0;p<out.size()/4;++p) {for(unsigned c=0;c<3;++c)out[p*4+c]=uint16_t(math.half_bits(rgb[p*3+c]));out[p*4+3]=0x3c00;}
            return out;
        };
        auto packed_color=half_rgba(color),packed_bb=half_rgba(bb);
        std::vector<uint16_t> packed_motion(motion.size());
        for(size_t i=0;i<motion.size();++i)packed_motion[i]=uint16_t(math.half_bits(motion[i]));
        session->prepare_plane(session->host_color,DXGI_FORMAT_R16G16B16A16_FLOAT,false);
        session->prepare_plane(session->host_output,DXGI_FORMAT_R16G16B16A16_FLOAT,true);
        if(!motion.empty())session->prepare_plane(session->host_motion,DXGI_FORMAT_R16G16_FLOAT,false);
        if(!mask.empty())session->prepare_plane(session->host_mask,DXGI_FORMAT_R32G32B32A32_FLOAT,false);
        if(!ui.empty())session->prepare_plane(session->host_ui,DXGI_FORMAT_R32G32B32A32_FLOAT,false);
        if(!alpha.empty())session->prepare_plane(session->host_alpha,DXGI_FORMAT_R32_FLOAT,false);
        if(!bb.empty())session->prepare_plane(session->host_backbuffer,DXGI_FORMAT_R16G16B16A16_FLOAT,false);
        if(!field.empty())session->prepare_plane(session->host_distortion,DXGI_FORMAT_R32G32B32A32_FLOAT,false);
        const bool measure_call=session->measure_gpu||session->pipeline.monitoring!=0||session->neural_budget_enabled();
        const auto started=measure_call?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        try {
            session->begin();
            session->upload(session->host_color,packed_color.data(),w*8,w*8);
            if(!motion.empty())session->upload(session->host_motion,packed_motion.data(),w*4,w*4);
            if(!mask.empty())session->upload(session->host_mask,mask.data(),w*16,w*16);
            if(!ui.empty())session->upload(session->host_ui,ui.data(),w*16,w*16);
            if(!alpha.empty())session->upload(session->host_alpha,alpha.data(),w*4,w*4);
            if(!bb.empty())session->upload(session->host_backbuffer,packed_bb.data(),w*8,w*8);
            if(!field.empty())session->upload(session->host_distortion,field.data(),w*16,w*16);
            session->submit();session->wait(session->submitted);
            const auto state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            PhoenixNativeFrameV2 frame{};frame.struct_bytes=sizeof(frame);frame.version=2;
            phoenix_native_default_controls(&frame.controls);
            frame.controls.reset=metadata.value("reset",false);frame.controls.local_tone=controls.tone;
            frame.controls.local_structure=controls.structure;frame.controls.skin_structure=controls.skin;
            frame.controls.auto_mask=controls.auto_mask;frame.controls.inverted_depth=metadata.value("reversed_depth",false);
            frame.intensity=controls.intensity;frame.style=controls.style;
            frame.color={session->host_color.image.Get(),state,0};frame.output={session->host_output.image.Get(),state,0};
            if(!motion.empty())frame.motion={session->host_motion.image.Get(),state,0};
            if(!mask.empty())frame.control_mask={session->host_mask.image.Get(),state,0};
            if(!ui.empty())frame.ui={session->host_ui.image.Get(),state,0};
            if(!alpha.empty())frame.ui_alpha={session->host_alpha.image.Get(),state,0};
            if(!bb.empty())frame.backbuffer={session->host_backbuffer.image.Get(),state,0};
            if(!field.empty())frame.distortion={session->host_distortion.image.Get(),state,0};
            session->frame(frame.color.resource,state,frame.motion.resource,state,frame.output.resource,state,&frame.controls,&frame);
            session->begin();auto& plane=session->host_output;
            phoenix_native::transition(session->command.Get(),plane.image.Get(),state,D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION from{plane.image.Get(),D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0},
                to{plane.transfer.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,plane.footprint};
            session->command->CopyTextureRegion(&to,0,0,0,&from,nullptr);
            phoenix_native::transition(session->command.Get(),plane.image.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,state);
            session->submit();session->wait(session->submitted);
            void* mapped=nullptr;D3D12_RANGE range{0,size_t(plane.bytes)},empty{};
            hr(plane.transfer->Map(0,&range,&mapped));
            for(unsigned y=0;y<h;++y) {
                const auto* row=reinterpret_cast<const uint16_t*>(static_cast<const char*>(mapped)+plane.footprint.Offset+size_t(y)*plane.footprint.Footprint.RowPitch);
                auto* dst=reinterpret_cast<unsigned char*>(base.output.data)+size_t(y)*output_stride;
                for(unsigned x=0;x<w;++x)for(unsigned c=0;c<3;++c) {
                    const float v=math.half_value(row[x*4+c]);std::memcpy(dst+(size_t(x)*3+c)*sizeof(float),&v,sizeof(v));
                }
            }
            plane.transfer->Unmap(0,&empty);
            session->call_ms=measure_call?std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count():0;
        } catch(...) {session->poisoned=true;throw;}
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
        const bool measure_call=session->measure_gpu||session->pipeline.monitoring!=0||session->neural_budget_enabled();
        const auto started = measure_call?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
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
            session->call_ms = measure_call?std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - started)
                                   .count():0;
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
int phoenix_native_reconfigure(PhoenixNativeSession** session,const char* config) {
    return protect([&] {
        require(session&&*session&&config,"Missing native session/configuration");
        auto* previous=*session;previous->check_thread();
        auto replacement=std::make_unique<PhoenixNativeSession>(Json::parse(config),previous->device.Get(),previous->queue.Get());
        previous->drain();delete previous;*session=replacement.release();
    });
}
int phoenix_native_set_pipeline(PhoenixNativeSession* session, const char* options) {
    return protect([&] {
        require(session && options, "Missing session/pipeline settings");
        session->check_thread(); require(!session->poisoned, "Native session failed");
        auto candidate = phoenix_pipeline::options_from_json(Json::parse(options), session->pipeline);
        require(!candidate.chain_enabled || candidate.scaler != 2 ||
            !candidate.enabled[unsigned(phoenix_pipeline::Stage::Scale)], "RTX VSR is deferred; choose an available scaling provider");
        unsigned planned_w=session->width,planned_h=session->height;
        for(const auto& step:phoenix_pipeline::plan(candidate,session->width,session->height,session->output_width,session->output_height)){
            if(step.stage!=phoenix_pipeline::Stage::Scale)continue;
            if(candidate.scaler==3)require(step.width==planned_w*2&&step.height==planned_h*2,"Tiny reconstruction requires 2x output geometry");
            if(candidate.scaler==1)require(step.width>=planned_w&&step.height>=planned_h&&step.width<=planned_w*2&&step.height<=planned_h*2,"NIS supports 1x to 2x per axis");
            planned_w=step.width;planned_h=step.height;
        }
        session->drain();
        const bool measure = candidate.monitoring == int(phoenix_pipeline::Monitoring::Detailed);
        if (measure && !session->queries) {
            hr(session->queue->GetTimestampFrequency(&session->frequency));
            require(session->frequency != 0, "GPU timestamp frequency unavailable");
            D3D12_QUERY_HEAP_DESC description{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 0};
            hr(session->device->CreateQueryHeap(&description, IID_PPV_ARGS(&session->queries)));
            session->timestamps = phoenix_native::buffer(session->device.Get(), 16,
                D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        }
        if (!measure) { session->queries.Reset(); session->timestamps.Reset(); session->gpu_ms = 0; }
        if (session->pipeline.motion_source != candidate.motion_source ||
            session->pipeline.flow_height != candidate.flow_height ||
            session->pipeline.motion_enabled != candidate.motion_enabled) {
            session->motion_provider.reset(); session->history = false;
        }
        const auto& prior=session->pipeline;
        const bool graph_changed=prior.chain_enabled!=candidate.chain_enabled||prior.order!=candidate.order||prior.enabled!=candidate.enabled||
            prior.scaler!=candidate.scaler||prior.fusion!=candidate.fusion||prior.fusion_layers!=candidate.fusion_layers||
            prior.fusion_second_scale!=candidate.fusion_second_scale||prior.fusion_third_scale!=candidate.fusion_third_scale;
        const bool color_changed=prior.color_brightness!=candidate.color_brightness||prior.color_contrast!=candidate.color_contrast||
            prior.color_saturation!=candidate.color_saturation||prior.color_gamma!=candidate.color_gamma;
        if(graph_changed||color_changed||prior.priority!=candidate.priority){session->graph.reset();session->source_discontinuity.clear();session->graph_owner=true;}
        session->configuration["pipeline"]=phoenix_pipeline::options_json(candidate);
        session->pipeline = candidate; session->measure_gpu = measure;
        session->requested_pipeline=candidate;session->latency_policy.reset();
        session->gpu_sample_interval = unsigned(candidate.sample_interval);
        session->gpu_sampled_last_frame = false;
    });
}
int phoenix_native_set_latency(PhoenixNativeSession* session,double source_age_ms,double presentation_tail_ms){
    return protect([&]{
        require(session&&std::isfinite(source_age_ms)&&source_age_ms>=0&&source_age_ms<=60000&&
            std::isfinite(presentation_tail_ms)&&presentation_tail_ms>=0&&presentation_tail_ms<=60000,"Invalid caller latency sample");
        session->check_thread();require(!session->poisoned,"Native session failed");
        const bool enabled=session->pipeline.monitoring!=0;
        session->pending_source_age_ms=enabled?source_age_ms:0;
        session->previous_presentation_tail_ms=enabled?presentation_tail_ms:0;
        session->caller_latency_pending=enabled;
    });
}
int phoenix_native_set_frame_info(PhoenixNativeSession* session, const PhoenixNativeFrameInfo* info) {
    return protect([&] {
        require(session && info && info->struct_bytes == sizeof(*info) && info->version == 1 &&
            info->engine_motion_valid <= 1 && !info->reserved, "Invalid native frame identity");
        session->check_thread(); require(!session->poisoned, "Native session failed");
        session->next_frame_info = *info; session->frame_info_pending = true;
    });
}
int phoenix_native_set_tuning(PhoenixNativeSession* session,const char* text){
    return protect([&]{
        require(session&&text,"Missing native neural tuning");session->check_thread();
        require(!session->poisoned,"Native session failed");
        const auto candidate=phoenix_pipeline::Tuning::read(Json::parse(text));
        // Frontends submit their current recipe after initialization too. An
        // identical request must preserve the executor and temporal histories.
        if(session->configuration.contains("tuning")&&candidate==session->tuning&&candidate==session->requested_tuning)return;
        session->drain();
        const bool structural=!session->graph_owner||candidate.passes!=session->tuning.passes||candidate.repeats!=session->tuning.repeats||
            candidate.neural_scale!=session->tuning.neural_scale||candidate.blocks!=session->tuning.blocks;
        const auto previous_tuning=session->tuning;const auto previous_configuration=session->configuration;
        const auto previous_graph=session->graph;const bool previous_owner=session->graph_owner;
        session->tuning=candidate;session->configuration["tuning"]=candidate.json();session->graph_owner=true;
        try{if(structural)session->initialize_graph();}
        catch(...){session->tuning=previous_tuning;session->configuration=previous_configuration;session->graph=previous_graph;session->graph_owner=previous_owner;throw;}
        if(structural){session->source_discontinuity.clear();session->network.reset();session->dispatch.reset();session->input_color.Reset();session->input_motion.Reset();session->resize.reset();}
        session->requested_tuning=candidate;session->latency_policy.reset();session->rebuild_graph=false;
    });
}
size_t phoenix_native_status(PhoenixNativeSession* session, char* destination, size_t capacity) {
    size_t result = 0;
    protect([&] {
        require(session != nullptr, "Null native session");
        session->check_thread();
        const auto portable_stages=session->portable_stage_reports();
        auto inference_pids=Json::array();
        for(const auto& stage:portable_stages){
            const auto pid=stage.value("inference_pid",0u);
            if(pid&&std::find(inference_pids.begin(),inference_pids.end(),Json(pid))==inference_pids.end())inference_pids.push_back(pid);
        }
        if(!session->portable&&!session->spatial_only)inference_pids.push_back(GetCurrentProcessId());
        const auto text =
            Json{{"backend", session->spatial_only ? "spatial_d3d12" : session->portable?"portable_pipeline_d3d12":"native_rtx"},
                 {"processing_host_pid",GetCurrentProcessId()},
                 {"inference_pid",inference_pids.empty()?Json(0):inference_pids.front()},
                 {"inference_pids",inference_pids},
                 {"portable_neural_stages",portable_stages},
                 {"model_id", session->spatial_only ? "none" : session->weight_source.model_id},
                 {"model_architecture", session->spatial_only ? "none" : std::string(session->weight_source.model().architecture)},
                 {"model_resource", session->spatial_only ? "none" : std::string(session->weight_source.model().resource)},
                 {"width", session->width},
                 {"height", session->height},
                 {"output_width", session->output_width},
                 {"output_height", session->output_height},
                 {"neural_width", session->neural_width},
                 {"neural_height", session->neural_height},
                 {"gpu_processing_ms", session->gpu_ms},
                 {"complete_call_ms", session->call_ms},
                 {"source_to_sdk_completion_ms",session->completion_age_ms},{"caller_latency_available",session->completed_caller_latency},
                 {"gpu_timing_scope",
                  session->portable?"portable backend network samples; complete_call_ms includes host transfers and D3D12 pipeline stages":
                  "model plus presentation/unwarp/resize/copy; excludes input canonicalization/preparation and host transfers"},
                 {"frames", session->completed_frames},
                 {"pipeline", phoenix_pipeline::options_json(session->pipeline)},
                 {"tuning",session->tuning.json()},
                 {"executed_chain",session->graph?session->graph->executed:Json::array()},
                 {"graph_neural_evaluations",session->graph_evaluations},
                 {"requested_tuning",session->requested_tuning.json()},
                 {"completed_tuning",session->completed_tuning},{"completed_pipeline",session->completed_pipeline},
                 {"pipeline_telemetry",session->pipeline_telemetry},
                 {"independent_temporal_stages",session->graph?session->graph->networks.size():unsigned(bool(session->network))},
                 {"chain_execution",session->graph_owner?"recorded stages":"original executor (equivalent stages folded)"},
                 {"motion_provider", session->motion_status},
                 {"gpu_monitoring_enabled", session->measure_gpu},
                 {"gpu_sampled_last_frame", session->gpu_sampled_last_frame},
                 {"additional_scaler", session->scaler_status},
                 {"tiny_workspace_bytes", session->tiny ? session->tiny->workspace_bytes() : 0},
                 {"native_frame_api_versions",{1,2}},
                 {"advanced_extent_contract","equal source/network/output"},
                 {"appearance_profile","recovered_3108"},
                 {"advanced_api_used",session->last_advanced},
                 {"ui_composition_used",session->last_ui},
                 {"effective_style",session->last_style},
                 {"effective_intensity",session->last_intensity},
                 {"packed_weights_sha256",session->packed_weights_sha256},
                 {"control_mask_used",session->had_mask},
                 {"distortion_used",session->had_distortion},
                 {"vendor_feature_parity_validated",false},
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
