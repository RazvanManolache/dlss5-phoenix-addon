// Production native RTX execution: creates D3D12 buffers, binds model resources and records kernel chains.
// This path consumes the launch contract; portable graph operators use a different backend interface.

#pragma once
#include "../model/bundle_loader.h"
#include "../runtime/kernel_module.h"
#include "../support/network_policy.h"
#include "../support/controls.h"
#include "../support/memory.h"
#include <algorithm>
#include <cstring>
#include <source_location>
#include <sstream>
#include <iomanip>

// Candidate-only, same-queue recorder. Callers retain this object and its inputs
// until their submission fence completes. No NGX/DLSS host or reference replay.
namespace phoenix_native {
inline constexpr const wchar_t* nvapi_library_name() {
    return sizeof(void*) == 8 ? L"nvapi64.dll" : L"nvapi.dll";
}
using Microsoft::WRL::ComPtr;
using phoenix_bundle::Json;
using phoenix_bundle::require;
inline void hr(HRESULT value, const std::source_location& location = std::source_location::current()) {
    if (FAILED(value)) {
        std::ostringstream message;
        message << "phoenix D3D12 failure 0x" << std::hex << std::uppercase << std::setw(8)
                << std::setfill('0') << uint32_t(value) << " in " << location.function_name()
                << " (line " << std::dec << location.line() << ')';
        throw std::runtime_error(message.str());
    }
}
inline void nv(NvAPI_Status value) {
    require(value == NVAPI_OK, "phoenix NVAPI failure");
}
struct Api {
    HMODULE library = nullptr;
    phoenix_kernel::Api modules;
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) dispatch = nullptr;
    decltype(&NvAPI_D3D12_GetCudaMergedTextureSamplerObject) texture = nullptr;
    decltype(&NvAPI_D3D12_GetCudaSurfaceObject) surface = nullptr;
    Api() {
        library = LoadLibraryExW(nvapi_library_name(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        require(library != nullptr, "NVAPI unavailable");
        auto query = reinterpret_cast<void*(__cdecl*)(unsigned)>(
            GetProcAddress(library, "nvapi_QueryInterface"));
        require(query != nullptr, "NVAPI query unavailable");
        auto initialize = reinterpret_cast<NvAPI_Status(__cdecl*)()>(query(0x0150e828));
        require(initialize != nullptr, "NVAPI init unavailable");
        nv(initialize());
        modules = {reinterpret_cast<decltype(modules.create_module)>(query(0xad1a677d)),
                   reinterpret_cast<decltype(modules.create_function)>(query(0xe2436e22)),
                   reinterpret_cast<decltype(modules.destroy_function)>(query(0xdf295ea6)),
                   reinterpret_cast<decltype(modules.destroy_module)>(query(0x41c65285)),
                   reinterpret_cast<decltype(modules.fatbin_supported)>(query(0x70c07832))};
        modules.validate();
        dispatch = reinterpret_cast<decltype(dispatch)>(query(0x24973538));
        texture = reinterpret_cast<decltype(texture)>(query(0x329fe6e0));
        surface = reinterpret_cast<decltype(surface)>(query(0x48f5b2ee));
        require(dispatch && texture && surface, "NVAPI binding/dispatch unavailable");
    }
    // CPU-test construction only: loads nothing and binds nothing. Every
    // entry stays null, so any use faults instead of silently dispatching.
    // Production always uses Api().
    struct NoLoad {};
    explicit Api(NoLoad) noexcept {}
    Api(const Api&) = delete;
    ~Api() {
        if (library)
            FreeLibrary(library);
    }
};
// Build module specifications from the bundle's named kernel records. Clear
// and copy helpers are supplied elsewhere, so they are excluded from this list.
inline std::vector<phoenix_kernel::Spec> catalog(const phoenix_bundle::Bundle& bundle) {
    std::vector<phoenix_kernel::Spec> specs;
    for (const auto& item : bundle.kernels.items()) {
        if (item.key() == "cc_cb_clear" || item.key() == "cg2r_copy_kernel")
            continue;
        const auto& record = item.value();
        auto block_extent = record.at("block").get<std::array<unsigned, 3>>();
        specs.push_back({item.key(), record.at("binary"), record.at("binary_sha256"),
                         record.at("entry"), record.at("parameter_bytes"), block_extent[0],
                         block_extent[1], block_extent[2],
                         record.value("large_input_entry", std::string()),
                         record.value("large_input_min_pixels", uint64_t(0))});
    }
    return specs;
}
struct BufferAllocationError : std::runtime_error {
    HRESULT code;
    static std::string describe(HRESULT result, uint64_t bytes, D3D12_HEAP_TYPE heap) {
        std::ostringstream message;
        message << "Native D3D12 buffer allocation failed: bytes=" << bytes
                << ", heap=" << unsigned(heap) << ", HRESULT=0x" << std::hex << uint32_t(result);
        return message.str();
    }
    BufferAllocationError(HRESULT result, uint64_t bytes, D3D12_HEAP_TYPE heap)
        : std::runtime_error(describe(result, bytes, heap)), code(result) {}
};
inline ComPtr<ID3D12Resource> buffer(ID3D12Device* device, uint64_t bytes,
                                     D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = heap_type;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (heap_type == D3D12_HEAP_TYPE_DEFAULT)
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> result;
    const HRESULT created = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                             state, nullptr, IID_PPV_ARGS(&result));
    if (FAILED(created))
        throw BufferAllocationError(created, bytes, heap_type);
    return result;
}
inline ComPtr<ID3D12Resource> image(ID3D12Device* device, unsigned width, unsigned height,
                                    DXGI_FORMAT format, D3D12_RESOURCE_STATES state,
                                    bool writable = true) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Format = format;
    if (writable)
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> result;
    hr(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                       IID_PPV_ARGS(&result)));
    return result;
}
inline void transition(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource,
                       D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (before == after)
        return;
    D3D12_RESOURCE_BARRIER transition{};
    transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    transition.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    cmd->ResourceBarrier(1, &transition);
}
inline void uav(ID3D12GraphicsCommandList* cmd) {
    D3D12_RESOURCE_BARRIER transition{};
    transition.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    cmd->ResourceBarrier(1, &transition);
}
// A wrapped engine answers ID3D12DeviceChild::GetDevice with different objects
// depending on the child: measured live, its command lists report one device
// object and its resources another, and no pointer equality can satisfy both.
// Ownership is therefore established by construction - an allocation made
// through the reported device must report the owner - which reduces to plain
// identity whenever there is no wrapper, as in the standalone host.
inline bool device_identity(ID3D12Device* left, ID3D12Device* right) {
    if (!left || !right)
        return false;
    ComPtr<IUnknown> a, transition;
    return SUCCEEDED(left->QueryInterface(IID_PPV_ARGS(&a))) &&
           SUCCEEDED(right->QueryInterface(IID_PPV_ARGS(&transition))) &&
           a.Get() == transition.Get();
}
inline bool allocation_reports(ID3D12Device* through, ID3D12Device* expected) {
    if (!through || !expected)
        return false;
    try {
        auto probe = image(through, 4, 4, DXGI_FORMAT_R16G16B16A16_FLOAT,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, false);
        ComPtr<ID3D12Device> probe_owner;
        if (FAILED(probe->GetDevice(IID_PPV_ARGS(&probe_owner))))
            return false;
        return device_identity(probe_owner.Get(), expected);
    } catch (...) {
        return false;
    }
}
// Two handles denote the same underlying device when an allocation made
// through either one is reported as owned by the other. Both directions are
// tried because a layer may forward allocations downwards while the runtime
// reports ownership upwards; identical pointers short-circuit, which is the
// only case the standalone host ever reaches.
inline bool device_owns(ID3D12Device* reported, ID3D12Device* owner) {
    if (device_identity(reported, owner))
        return true;
    if (!reported || !owner)
        return false;
    return allocation_reports(reported, owner) || allocation_reports(owner, reported);
}
// Optional recording-progress sink. A host that records into an engine's own
// command list points this at its evidence writer, so a step that never
// returns still leaves a record of how far the recording reached. The
// standalone host leaves it unset and pays nothing.
inline void (*record_progress)(const char*) = nullptr;
inline void note(const char* stage) noexcept {
    if (record_progress) {
        try {
            record_progress(stage);
        } catch (...) {
        }
    }
}
// Optional natural-dispatch-group marker. A host that wants to know where the
// frame's time goes points this at its own timestamp writer; it is called with
// the recording command list once per captured group boundary, in order. The
// grouping, the launches and the barriers are identical either way.
inline void (*group_marker)(ID3D12GraphicsCommandList*, unsigned) = nullptr;
inline void mark_group(ID3D12GraphicsCommandList* cmd, unsigned group) noexcept {
    if (group_marker) {
        try {
            group_marker(cmd, group);
        } catch (...) {
        }
    }
}
inline void validate_input(ID3D12Device* device, ID3D12Resource* input, unsigned width,
                           unsigned height, bool motion) {
    require(input != nullptr, "missing input");
    auto description = input->GetDesc();
    require(description.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
                description.Width == width && description.Height == height &&
                description.DepthOrArraySize == 1 && description.MipLevels == 1 &&
                description.SampleDesc.Count == 1 &&
                !(description.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE),
            "unsupported input geometry/flags");
    require(description.Format == DXGI_FORMAT_R16G16B16A16_FLOAT ||
                description.Format == DXGI_FORMAT_R32G32B32A32_FLOAT ||
                (motion && (description.Format == DXGI_FORMAT_R16G16_FLOAT ||
                            description.Format == DXGI_FORMAT_R32G32_FLOAT)),
            "unsupported input format");
    ComPtr<ID3D12Device> owner;
    hr(input->GetDevice(IID_PPV_ARGS(&owner)));
    require(device_owns(owner.Get(), device), "input belongs to another device");
}
struct Step {
    NVAPI_CU_KERNEL_LAUNCH_PARAMS launch{};
    std::vector<uint8_t> bytes;
    bool barrier_after = false;
};
class Network {
    // The opt-in native diagnostic executable inspects workspace snapshots and
    // can stop a prefix for fault isolation; production callers cannot do so.
    friend struct NetworkDiagnostics;
    // Refinement passes run serially on one queue. Their temporary tensors can
    // share storage; each pass still owns both of its temporal history images.
    // The barrier before each counter clear orders all preceding workspace use.
    struct SerialWorkspace {
        ComPtr<ID3D12Resource> resource;
        ComPtr<ID3D12CommandQueue> queue;
    };
    struct ImmutableModel {
        ComPtr<ID3D12Resource> weights;
        std::vector<ComPtr<ID3D12Resource>> uploads;
        ComPtr<ID3D12Fence> upload_complete;
        ID3D12GraphicsCommandList* initialization_command = nullptr;
        bool initialized = false, upload_submitted = false;
    };
    std::shared_ptr<ImmutableModel> model_storage_;
    std::shared_ptr<SerialWorkspace> serial_workspace_;
    std::shared_ptr<memory::Generation> accounting_;
    const void* weight_identity_ = nullptr;
    Api& api_;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12Resource> input_, motion_, weights_, workspace_, output_[2];
    ComPtr<ID3D12DescriptorHeap> views_, samplers_, clear_cpu_;
    D3D12_GPU_DESCRIPTOR_HANDLE clear_gpu_{};
    std::vector<std::unique_ptr<phoenix_kernel::Module>> modules_;
    std::array<std::vector<Step>, 4> programs_;
    std::array<std::array<std::vector<uint8_t>, 2>, 4> image_arguments_;
    std::vector<unsigned> group_ends_;
    phoenix_addon::Sampling sampling_;
    bool history_ = false, failed_ = false, pending_ = false;
    ID3D12GraphicsCommandList* pending_command_ = nullptr;
    ComPtr<ID3D12CommandQueue> queue_;
    void track_initial_upload() noexcept {
        auto& model = *model_storage_;
        if (!model.uploads.empty() && !model.upload_submitted) {
            // One fence covers the submitted list containing the initial copy
            // and every serial refinement pass. A failed signal retains staging.
            model.upload_submitted = SUCCEEDED(queue_->Signal(model.upload_complete.Get(), 1));
        }
    }
    void retire_initial_upload() noexcept {
        auto& model = *model_storage_;
        // UINT64_MAX means device removal, not successful completion.
        if (model.upload_submitted && model.upload_complete &&
            model.upload_complete->GetCompletedValue() == 1) {
            model.uploads.clear();
            model.upload_complete.Reset();
            model.initialization_command = nullptr;
        }
    }
    // Accepted once per distinct reported device, so the ownership probe is not
    // repeated for every recorded frame.
    mutable ComPtr<ID3D12Device> accepted_command_device_, accepted_queue_device_;
    const std::thread::id thread_ = std::this_thread::get_id();
    unsigned next_ = 0;
    uint64_t frames_ = 0, resets_ = 0;
    // Create the SRV/sampler pair at matching descriptor slots, then ask NVAPI
    // for the merged CUDA texture handle used by the native kernel argument words.
    uint64_t texture_handle(ID3D12Resource* image, unsigned slot, bool linear) {
        const auto view_increment =
            device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const auto sampler_increment =
            device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        auto view = views_->GetCPUDescriptorHandleForHeapStart();
        view.ptr += slot * view_increment;
        auto sampler = samplers_->GetCPUDescriptorHandleForHeapStart();
        sampler.ptr += slot * sampler_increment;
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = image->GetDesc().Format;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView(image, &srv, view);
        D3D12_SAMPLER_DESC sampler_description{};
        sampler_description.Filter =
            linear ? D3D12_FILTER_MIN_MAG_MIP_LINEAR : D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler_description.AddressU = sampler_description.AddressV = sampler_description.AddressW =
            D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler_description.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        sampler_description.MaxLOD = D3D12_FLOAT32_MAX;
        sampler_description.MaxAnisotropy = 1;
        device_->CreateSampler(&sampler_description, sampler);
        NVAPI_D3D12_GET_CUDA_MERGED_TEXTURE_SAMPLER_OBJECT_PARAMS texture_parameters{};
        texture_parameters.structSizeIn = texture_parameters.structSizeOut =
            sizeof(texture_parameters);
        texture_parameters.pDevice = device_.Get();
        texture_parameters.texDesc = view;
        texture_parameters.smpDesc = sampler;
        nv(api_.texture(&texture_parameters));
        return texture_parameters.textureHandle;
    }

  public:
    uint64_t control_mask_handle = 0;
    // Bind an owned stable texture before recording. The caller must retire
    // earlier submissions before changing this descriptor.
    void bind_control_mask(ID3D12Resource* control_mask) {
        require(control_mask&&!pending_,"Control map binding during a pending recording");
        // First-use descriptor publication must not race earlier submissions.
        // This happens once when a mask is enabled, never on ordinary frames.
        if (queue_ && frames_) {
            ComPtr<ID3D12Fence> retired;
            hr(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&retired)));
            struct EventOwner { HANDLE value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
                ~EventOwner() { if (value) CloseHandle(value); } } event;
            require(event.value != nullptr, "Control map retirement event");
            hr(queue_->Signal(retired.Get(), 1));
            hr(retired->SetEventOnCompletion(1, event.value));
            require(WaitForSingleObject(event.value, 10000) == WAIT_OBJECT_0 &&
                    retired->GetCompletedValue() == 1, "Control map descriptor retirement failed");
        }
        const auto d=control_mask->GetDesc();
        ComPtr<ID3D12Device> owner;hr(control_mask->GetDevice(IID_PPV_ARGS(&owner)));
        // ReShade may expose a device wrapper while its textures report the
        // underlying owner. Prove the allocation relationship, as for the
        // network's other resources, rather than rejecting that valid wrapper.
        require(device_owns(device_.Get(),owner.Get())&&d.Dimension==D3D12_RESOURCE_DIMENSION_TEXTURE2D&&
            d.Width==width&&d.Height==height&&d.MipLevels==1&&d.SampleDesc.Count==1&&d.DepthOrArraySize==1&&
            d.Format==DXGI_FORMAT_R32G32B32A32_FLOAT&&!(d.Flags&D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE),"Control map texture contract");
        control_mask_handle=texture_handle(control_mask,7,false);
    }
    const unsigned width, height;
    std::shared_ptr<phoenix_kernel::Counters> counters =
        std::make_shared<phoenix_kernel::Counters>();
    Network(ID3D12Device* device, Api& api, const Json& frame, const phoenix_model::WeightView& packed,
            const std::vector<phoenix_kernel::Binary>& candidates, ID3D12Resource* color,
            ID3D12Resource* motion = nullptr, std::shared_ptr<memory::Generation> accounting = {},
            bool allow_gpu_upload = true, const Network* sequential_source = nullptr,
            ID3D12Resource* control_mask = nullptr)
        : accounting_(accounting), weight_identity_(packed.identity()), api_(api), device_(device),
          input_(color), motion_(motion), width(frame.at("width")), height(frame.at("height")) {
        if (accounting) {
            if (accounting->modules)
                counters = accounting->modules;
            else
                accounting->modules = counters;
        }
        const auto contract = preflight(frame, candidates);
        require(frame.at("launches").size() == contract.neural_launches + 2 && phoenix_bundle::has_barrier_policy(frame),
                "incomplete plan");
        require(candidates.size() == 41, "candidate catalog incomplete");
        validate_input(device, color, width, height, false);
        if (motion)
            validate_input(device, motion, width, height, true);
        require(!packed.empty() &&
                    packed.size() == frame.at("arenas").at("weights").at("bytes").get<size_t>(),
                "weights extent");
        if (sequential_source)
            require(device_identity(device, sequential_source->device_.Get()) &&
                        &api == &sequential_source->api_ && width == sequential_source->width &&
                        height == sequential_source->height &&
                        packed.identity() == sequential_source->weight_identity_ &&
                        accounting == sequential_source->accounting_,
                    "shared refinement resources require the same device, model and owner");
        std::unordered_map<std::string, NVDX_ObjectHandle> functions;
        for (const auto& binary : candidates) {
            auto module = std::make_unique<phoenix_kernel::Module>(device, api.modules, counters);
            nv(api.modules.create_module(device, binary.bytes.data(), NvU32(binary.bytes.size()),
                                         &module->module));
            ++counters->modules_created;
            // Larger inputs can benefit from more resident blocks, while the
            // same register limit loses on small inputs. Keep both compiled
            // choices in one module; only the selected function gets a handle.
            const auto& entry = phoenix_kernel::entry_for_resolution(
                binary.spec.entry, binary.spec.large_input_entry,
                binary.spec.large_input_min_pixels, width, height);
            nv(api.modules.create_function(device, module->module, entry.c_str(),
                                           &module->function));
            ++counters->functions_created;
            require(functions.emplace(binary.spec.name, module->function).second,
                    "duplicate module");
            modules_.push_back(std::move(module));
        }
        const auto extent = frame.at("arenas").at("workspace").at("bytes").get<uint64_t>();
        if (sequential_source) {
            require(sequential_source->workspace_->GetDesc().Width == extent,
                    "shared refinement workspace extent mismatch");
            serial_workspace_ = sequential_source->serial_workspace_;
            workspace_ = serial_workspace_->resource;
        } else {
            serial_workspace_ = std::make_shared<SerialWorkspace>();
            workspace_ = buffer(device, extent, D3D12_HEAP_TYPE_DEFAULT,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            serial_workspace_->resource = workspace_;
            memory::resource(device, workspace_.Get(), accounting, memory::Kind::workspace);
        }
        if (sequential_source) {
            // The model is immutable after its one initialization. Both mapped
            // and staged routes share its lifetime, copy state and retirement
            // fence; later passes never repeat a COPY_DEST transition.
            model_storage_ = sequential_source->model_storage_;
            require(model_storage_->weights->GetDesc().Width == packed.size(),
                    "shared refinement model extent mismatch");
            weights_ = model_storage_->weights;
        } else {
            model_storage_ = std::make_shared<ImmutableModel>();
            auto& model = *model_storage_;
            D3D12_FEATURE_DATA_D3D12_OPTIONS16 options{};
            bool mapped_gpu_weights =
                allow_gpu_upload &&
                SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &options,
                                                      sizeof(options))) &&
                options.GPUUploadHeapSupported;
            phoenix_model::WeightTransfer transfer(packed);
            auto fill_upload = [&](ID3D12Resource* target, size_t offset, size_t count,
                                   D3D12_HEAP_TYPE heap) {
                void* data = nullptr;
                D3D12_RANGE empty{};
                const HRESULT mapped = target->Map(0, &empty, &data);
                if (FAILED(mapped))
                    throw BufferAllocationError(mapped, count, heap);
                try {
                    transfer.copy_to(offset, data, count);
                } catch (...) {
                    target->Unmap(0, nullptr);
                    throw;
                }
                target->Unmap(0, nullptr);
            };
            if (mapped_gpu_weights) {
                try {
                    weights_ = buffer(device, packed.size(), D3D12_HEAP_TYPE_GPU_UPLOAD,
                                      D3D12_RESOURCE_STATE_GENERIC_READ);
                    fill_upload(weights_.Get(), 0, packed.size(), D3D12_HEAP_TYPE_GPU_UPLOAD);
                    model.initialized = true;
                } catch (const BufferAllocationError& error) {
                    if (error.code != E_OUTOFMEMORY)
                        throw;
                    // Feature support is not an allocation guarantee. A WOW64
                    // game can lack a model-sized CPU-visible BAR address range.
                    weights_.Reset();
                    mapped_gpu_weights = false;
                }
            }
            if (!mapped_gpu_weights) {
                weights_ = buffer(device, packed.size(), D3D12_HEAP_TYPE_DEFAULT,
                                  D3D12_RESOURCE_STATE_COPY_DEST);
                auto append_upload = [&](size_t offset, size_t count) {
                    auto staging = buffer(device, count, D3D12_HEAP_TYPE_UPLOAD,
                                          D3D12_RESOURCE_STATE_GENERIC_READ);
                    fill_upload(staging.Get(), offset, count, D3D12_HEAP_TYPE_UPLOAD);
                    memory::resource(device, staging.Get(), accounting, memory::Kind::upload);
                    model.uploads.push_back(std::move(staging));
                };
                // Keep the GPU weights contiguous while staging them through
                // small CPU mappings. These buffers exist only until the first
                // submitted copy fence completes, including shared refinements.
                constexpr size_t chunk_bytes = 4 * 1024 * 1024;
                for (size_t offset = 0; offset < packed.size();) {
                    const auto count = std::min(chunk_bytes, packed.size() - offset);
                    append_upload(offset, count);
                    offset += count;
                }
                hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                       IID_PPV_ARGS(&model.upload_complete)));
                memory::track(model.upload_complete.Get(), accounting, memory::Kind::fence);
            }
            transfer.finish();
            model.weights = weights_;
            memory::resource(device, weights_.Get(), accounting, memory::Kind::weights);
        }
        for (auto& out : output_) {
            out = image(device, width, height, DXGI_FORMAT_R16G16B16A16_FLOAT,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            memory::resource(device, out.Get(), accounting, memory::Kind::history);
        }
        D3D12_DESCRIPTOR_HEAP_DESC heap_description{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8,
                                                    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        hr(device->CreateDescriptorHeap(&heap_description, IID_PPV_ARGS(&views_)));
        memory::track(views_.Get(), accounting, memory::Kind::views);
        heap_description.NumDescriptors = 1;
        heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        hr(device->CreateDescriptorHeap(&heap_description, IID_PPV_ARGS(&clear_cpu_)));
        memory::track(clear_cpu_.Get(), accounting, memory::Kind::views);
        D3D12_UNORDERED_ACCESS_VIEW_DESC counter_view{};
        counter_view.Format = DXGI_FORMAT_R32_UINT;
        counter_view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        counter_view.Buffer.NumElements = contract.clear_words;
        auto counter_handle = views_->GetCPUDescriptorHandleForHeapStart();
        counter_handle.ptr +=
            6 * device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        device->CreateUnorderedAccessView(workspace_.Get(), nullptr, &counter_view, counter_handle);
        device->CreateUnorderedAccessView(workspace_.Get(), nullptr, &counter_view,
                                          clear_cpu_->GetCPUDescriptorHandleForHeapStart());
        clear_gpu_ = views_->GetGPUDescriptorHandleForHeapStart();
        clear_gpu_.ptr +=
            6 * device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        heap_description.NumDescriptors = 8;
        heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr(device->CreateDescriptorHeap(&heap_description, IID_PPV_ARGS(&samplers_)));
        memory::track(samplers_.Get(), accounting, memory::Kind::samplers);
        const uint64_t color_handle = texture_handle(color, 0, false);
        const uint64_t motion_handle = motion ? texture_handle(motion, 1, false) : 0;
        if (control_mask) {
            bind_control_mask(control_mask);
        }
        uint64_t history_handles[2], surfaces[2];
        for (unsigned iteration_index = 0; iteration_index < 2; ++iteration_index) {
            history_handles[iteration_index] =
                texture_handle(output_[iteration_index].Get(), 2 + iteration_index, true);
            auto view_handle = views_->GetCPUDescriptorHandleForHeapStart();
            view_handle.ptr += (4 + iteration_index) * device->GetDescriptorHandleIncrementSize(
                                                           D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            D3D12_UNORDERED_ACCESS_VIEW_DESC unordered_view{};
            unordered_view.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            unordered_view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            device->CreateUnorderedAccessView(output_[iteration_index].Get(), nullptr,
                                              &unordered_view, view_handle);
            NvU32 handle = 0;
            nv(api.surface(device, view_handle, &handle));
            surfaces[iteration_index] = handle;
        }
        const std::array<phoenix_launch_plan::Arena, 2> arenas{
            {{workspace_->GetGPUVirtualAddress(), extent},
             {weights_->GetGPUVirtualAddress(), packed.size()}}};
        for (unsigned target = 0; target < 2; ++target)
            for (unsigned recurrent = 0; recurrent < 2; ++recurrent) {
                auto& steps = programs_[target * 2 + recurrent];
                steps.reserve(contract.neural_launches);
                for (unsigned iteration_index = 1; iteration_index <= contract.neural_launches; ++iteration_index) {
                    const auto& source = frame.at("launches")[iteration_index];
                    auto plan =
                        phoenix_bundle::launch(source, [&](const Json& binding) -> uint64_t {
                            const std::string role = binding.at("role");
                            if (role == "color")
                                return color_handle;
                            if (role == "history_previous")
                                return recurrent ? history_handles[1 - target] : 0;
                            if (role == "motion")
                                return recurrent ? motion_handle : 0;
                            if (role == "neural_output_surface")
                                return surfaces[target];
                            throw std::runtime_error("unsupported non-null descriptor role: " +
                                                     role);
                        });
                    Step step;
                    step.bytes = phoenix_launch_plan::materialize(plan, arenas);
                    step.launch.hFunction = functions.at(source.at("name").get<std::string>());
                    step.launch.gridDim = {plan.grid[0], plan.grid[1], plan.grid[2]};
                    step.launch.blockDim = {plan.block[0], plan.block[1], plan.block[2]};
                    step.launch.paramSize = NvU32(step.bytes.size());
                    step.barrier_after = contract.barrier_after[iteration_index - 1];
                    steps.push_back(std::move(step));
                }
                for (auto& step : steps)
                    step.launch.pParams = step.bytes.data();
                image_arguments_[target * 2 + recurrent] = {steps.front().bytes,
                                                            steps.back().bytes};
            }
        group_ends_ = contract.group_ends;
    }
    Network(const Network&) = delete;
    uint64_t host_container_capacity_bytes() const {
        uint64_t bytes = modules_.capacity() * sizeof(decltype(modules_)::value_type) +
                         group_ends_.capacity() * sizeof(unsigned);
        for (const auto& program : programs_) {
            bytes += program.capacity() * sizeof(Step);
            for (const auto& step : program)
                bytes += step.bytes.capacity();
        }
        for (const auto& pair : image_arguments_)
            for (const auto& args : pair)
                bytes += args.capacity();
        return bytes;
    }
    // Inputs must already be NON_PIXEL_SHADER_RESOURCE on the caller's DIRECT
    // queue. The returned image remains in that state; restore it after external
    // copies. Queue order, not CPU idle, protects ping-pong and workspace reuse.
    void validate_command(ID3D12GraphicsCommandList* cmd, ID3D12CommandQueue* queue) const {
        require(std::this_thread::get_id() == thread_ && cmd && queue && !failed_ && !pending_ &&
                    SUCCEEDED(device_->GetDeviceRemovedReason()),
                "invalid/lost/pending network or recording thread");
        ComPtr<ID3D12Device> d;
        hr(cmd->GetDevice(IID_PPV_ARGS(&d)));
        if (d.Get() != accepted_command_device_.Get()) {
            require(device_owns(d.Get(), device_.Get()), "wrong command device");
            accepted_command_device_ = d;
        }
        require(cmd->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT, "direct queue required");
        hr(queue->GetDevice(IID_PPV_ARGS(&d)));
        if (d.Get() != accepted_queue_device_.Get()) {
            require(device_owns(d.Get(), device_.Get()), "wrong queue device");
            accepted_queue_device_ = d;
        }
        require(queue->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT, "wrong queue device/type");
        require(!queue_ || queue_.Get() == queue, "cross-queue history reuse rejected");
        require(!serial_workspace_->queue || serial_workspace_->queue.Get() == queue,
                "shared refinement workspace cannot cross queues");
    }
    // Comparison-only twin of validate_command above: identical except it
    // does not pin the caller to the construction thread. RE9 evaluates
    // its two NGX features on different threads (b4: frame 1 recorded on
    // one thread, frame 2's commit arrived on another and the strict form
    // killed the session), so the comparison path cannot promise
    // single-threaded use. The caller must externally serialize EVERY
    // access to this object instead -- the ASI holds g_mutex across each
    // whole record_comparison call, which subsumes the check's purpose
    // (detecting concurrent or aliased use). The claim path keeps the
    // strict form: its transaction/submission pairing is genuinely
    // thread-bound (see Session::settle).
    void validate_command_any_thread(ID3D12GraphicsCommandList* cmd,
                                     ID3D12CommandQueue* queue) const {
        require(cmd && queue && !failed_ && !pending_ &&
                    SUCCEEDED(device_->GetDeviceRemovedReason()),
                "invalid/lost/pending network");
        ComPtr<ID3D12Device> d;
        hr(cmd->GetDevice(IID_PPV_ARGS(&d)));
        if (d.Get() != accepted_command_device_.Get()) {
            require(device_owns(d.Get(), device_.Get()), "wrong command device");
            accepted_command_device_ = d;
        }
        require(cmd->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT, "direct queue required");
        hr(queue->GetDevice(IID_PPV_ARGS(&d)));
        if (d.Get() != accepted_queue_device_.Get()) {
            require(device_owns(d.Get(), device_.Get()), "wrong queue device");
            accepted_queue_device_ = d;
        }
        require(queue->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT, "wrong queue device/type");
        require(!queue_ || queue_.Get() == queue, "cross-queue history reuse rejected");
        require(!serial_workspace_->queue || serial_workspace_->queue.Get() == queue,
                "shared refinement workspace cannot cross queues");
    }
    // `any_thread` is the comparison path (see validate_command_any_thread):
    // the caller serializes externally. Defaults strict for the claim path.
    ID3D12Resource* record(ID3D12GraphicsCommandList* cmd, bool reset, uint32_t /*frame_label*/,
                           ID3D12CommandQueue* queue, bool any_thread = false) {
        if (any_thread)
            validate_command_any_thread(cmd, queue);
        else
            validate_command(cmd, queue);
        queue_ = queue;
        serial_workspace_->queue = queue;
        try {
            retire_initial_upload();
            auto& model = *model_storage_;
            require(model.uploads.empty() || !model.initialized || model.upload_submitted ||
                        model.initialization_command == cmd,
                    "shared model copy must be recorded in the same initial submission");
            if (!model.initialized) {
                uint64_t offset = 0;
                for (const auto& upload : model.uploads) {
                    const auto bytes = upload->GetDesc().Width;
                    cmd->CopyBufferRegion(weights_.Get(), offset, upload.Get(), 0, bytes);
                    offset += bytes;
                }
                require(offset == weights_->GetDesc().Width, "Incomplete initial model upload");
                transition(cmd, weights_.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                model.initialization_command = cmd;
                model.initialized = true;
            }
            // Reset or missing motion disables both motion and temporal history.
            const bool recurrent = temporal_reuse(history_, reset, bool(motion_));
            if (!recurrent)
                ++resets_;
            transition(cmd, output_[next_].Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            note("network_set_heaps");
            ID3D12DescriptorHeap* heaps[]{views_.Get(), samplers_.Get()};
            cmd->SetDescriptorHeaps(2, heaps);
            note("network_clear_workspace");
            // Synchronization counters start at the pending sentinel. Feature
            // tensors and model weights are not part of this clear operation.
            const UINT sentinel[4]{UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
            uav(cmd);
            cmd->ClearUnorderedAccessViewUint(clear_gpu_,
                                              clear_cpu_->GetCPUDescriptorHandleForHeapStart(),
                                              workspace_.Get(), sentinel, 0, nullptr);
            uav(cmd);
            note("network_first_dispatch");
            auto& program = programs_[next_ * 2 + unsigned(recurrent)];
            const auto& image_arguments = image_arguments_[next_ * 2 + unsigned(recurrent)];
            memcpy(program.front().bytes.data(), image_arguments[0].data(),
                   image_arguments[0].size());
            memcpy(program.back().bytes.data(), image_arguments[1].data(),
                   image_arguments[1].size());
            // Sampling advances after successful submission and resets on an
            // explicit history reset. Missing motion alone does not reset it.
            phoenix_addon::Controls controls;
            controls.reset = reset;
            const auto resolved = phoenix_addon::resolve_controls(
                controls, width, height, bool(motion_), history_, sampling_.begin(reset));
            phoenix_addon::patch_stem(program.front().bytes.data(), program.front().bytes.size(),
                                      resolved);
            phoenix_addon::patch_head(program.back().bytes.data(), program.back().bytes.size(),
                                      resolved);
            // Record launches individually, preserving their order and the
            // plan's group barriers. A large NVAPI chain was slower on RTX 5080
            // than these calls; adding a barrier after every launch also lost
            // performance. Kernels still use their tile publication counters
            // inside each group, and tensors are reused only after its barrier.
            unsigned launch_index = 0;
            for (unsigned group = 0; group < group_ends_.size(); ++group) {
                const unsigned end = group_ends_[group];
                for (; launch_index < end; ++launch_index)
                    nv(api_.dispatch(cmd, &program[launch_index].launch, 1));
                if (program[end - 1].barrier_after)
                    uav(cmd);
                mark_group(cmd, group);
            }
            auto* result = output_[next_].Get();
            transition(cmd, result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            next_ = 1 - next_;
            history_ = true;
            pending_ = true;
            pending_command_ = cmd;
            return result;
        } catch (...) {
            failed_ = true;
            sampling_.invalidate();
            throw;
        }
    }
    // Call only after successful submission to the queue passed to record().
    // Abandonment must invalidate/rebuild; another record cannot silently reuse it.
    void commit(ID3D12GraphicsCommandList* cmd) {
        require(std::this_thread::get_id() == thread_ && !failed_ && pending_ &&
                    pending_command_ == cmd,
                "invalid recording commit");
        sampling_.commit();
        track_initial_upload();
        pending_ = false;
        pending_command_ = nullptr;
        ++frames_;
    }
    // Comparison-only twin of commit above: identical except it does not
    // pin the caller to the construction thread. Same contract as
    // validate_command_any_thread: the caller serializes externally.
    void commit_any_thread(ID3D12GraphicsCommandList* cmd) {
        require(!failed_ && pending_ && pending_command_ == cmd, "invalid recording commit");
        sampling_.commit();
        track_initial_upload();
        pending_ = false;
        pending_command_ = nullptr;
        ++frames_;
    }
    uint64_t frames() const {
        return frames_;
    }
    uint64_t resets() const {
        return resets_;
    }
    uint64_t next_sampling_index() const {
        return sampling_.next();
    }
    const std::vector<unsigned>& group_ends() const noexcept {
        return group_ends_;
    }
    unsigned program_steps() const {
        return programs_.empty() ? 0u : static_cast<unsigned>(programs_.front().size());
    }
    ID3D12Resource* output(unsigned index) const {
        require(index < 2, "output index");
        return output_[index].Get();
    }
    void invalidate() noexcept {
        failed_ = true;
        sampling_.invalidate();
        pending_ = false;
        pending_command_ = nullptr;
    }
};
} // namespace production41
