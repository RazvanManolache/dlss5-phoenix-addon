// Offline single-instance inference smoke test and timing probe; never linked into the addon.
#include "../../src/native/runtime/texture_binding.h"
#include "../../src/native/support/direct.h"
#include "../../src/native/support/gpu_timing.h"
#include <dxgi1_6.h>
#include <iostream>
#include <chrono>
#include <psapi.h>
using namespace phoenix_addon;
namespace fs = std::filesystem;
#include "kernel_timing.h"
static phoenix::GpuTiming* active_timer = nullptr;
static std::string digest(const std::vector<uint8_t>& bytes) {
    unsigned char hash[32];
    char hex[65]{};
    check(BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0, const_cast<PUCHAR>(bytes.data()),
                     ULONG(bytes.size()), hash, 32) >= 0,
          "hash");
    for (unsigned i = 0; i < 32; ++i)
        sprintf_s(hex + i * 2, 3, "%02x", unsigned(hash[i]));
    return hex;
}
static unsigned debug_errors(ID3D12Device* device) {
    ComPtr<ID3D12InfoQueue> q;
    phoenix_native::hr(device->QueryInterface(IID_PPV_ARGS(&q)));
    unsigned errors = 0;
    for (UINT64 i = 0; i < q->GetNumStoredMessages(); ++i) {
        SIZE_T n = 0;
        q->GetMessage(i, nullptr, &n);
        std::vector<uint8_t> b(n);
        auto* m = reinterpret_cast<D3D12_MESSAGE*>(b.data());
        q->GetMessage(i, m, &n);
        if (m->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
            ++errors;
            std::cerr << m->pDescription << '\n';
        }
    }
    q->ClearStoredMessages();
    return errors;
}
static Json measure_memory(IDXGIAdapter1* adapter) {
    PROCESS_MEMORY_COUNTERS_EX c{};
    c.cb = sizeof(c);
    check(GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&c),
                               sizeof(c)),
          "process memory");
    Json result = {{"host",
                    {{"private_bytes", c.PrivateUsage},
                     {"peak_private_bytes", c.PeakPagefileUsage},
                     {"working_set_bytes", c.WorkingSetSize},
                     {"peak_working_set_bytes", c.PeakWorkingSetSize}}}};
    if (adapter) {
        ComPtr<IDXGIAdapter3> a;
        phoenix_native::hr(adapter->QueryInterface(IID_PPV_ARGS(&a)));
        DXGI_QUERY_VIDEO_MEMORY_INFO local{}, shared{};
        phoenix_native::hr(a->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local));
        phoenix_native::hr(
            a->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &shared));
        result["device_local_bytes"] = local.CurrentUsage;
        result["device_nonlocal_bytes"] = shared.CurrentUsage;
    }
    return result;
}
struct Payload final : phoenix_device::Payload {
    phoenix_native::memory::Ledger memory;
    phoenix_native::memory::Ledger harness_memory;
    std::shared_ptr<phoenix_native::memory::Generation> allocation;
    std::shared_ptr<phoenix_native::memory::Generation> harness_allocation;
    Texture color, motion, readback[1];
    FrameDispatch dispatch[1];
    std::unique_ptr<phoenix_native::Network> network[1];
    Payload(ID3D12Device* device, const Json& frame, const phoenix_model::WeightView& weights,
            const std::vector<phoenix_kernel::Binary>& candidate, const std::vector<uint8_t>& input,
            const std::vector<uint8_t>& vectors, bool allow_gpu_upload) {
        const unsigned w = frame.at("width"), h = frame.at("height");
        check(input.size() == size_t(w) * h * 8 && vectors.size() == size_t(w) * h * 4,
              "captured FP16 input sizes");
        color = texture(device, w, h, 16, false);
        motion = texture(device, w, h, 16, false, 2);
        allocation = memory.begin(w, h);
        harness_allocation = harness_memory.begin(w, h);
        phoenix_native::memory::resource(device, color.resource.Get(), allocation,
                                         phoenix_native::memory::Kind::color);
        phoenix_native::memory::resource(device, motion.resource.Get(), allocation,
                                         phoenix_native::memory::Kind::motion);
        for (unsigned kind = 0; kind < 2; ++kind) {
            auto& t = kind ? motion : color;
            phoenix_native::memory::resource(device, t.transfer.Get(), harness_allocation,
                                             phoenix_native::memory::Kind::upload);
            void* map = nullptr;
            D3D12_RANGE empty{};
            phoenix_native::hr(t.transfer->Map(0, &empty, &map));
            memset(map, 0, size_t(t.bytes));
            for (unsigned y = 0; y < h; ++y) {
                auto* row = static_cast<uint8_t*>(map) + t.footprint.Offset +
                            y * t.footprint.Footprint.RowPitch;
                if (!kind)
                    memcpy(row, input.data() + size_t(y) * w * 8, size_t(w) * 8);
                else
                    memcpy(row, vectors.data() + size_t(y) * w * 4, size_t(w) * 4);
            }
            t.transfer->Unmap(0, nullptr);
            if (kind)
                continue;
            readback[kind] = texture(device, w, h, 16, true);
            // The offline probe owns these resources; the addon does not read
            // every output back to the CPU. Keep their accounting separate.
            phoenix_native::memory::resource(device, readback[kind].resource.Get(),
                                             harness_allocation,
                                             phoenix_native::memory::Kind::history);
            phoenix_native::memory::resource(device, readback[kind].transfer.Get(),
                                             harness_allocation,
                                             phoenix_native::memory::Kind::upload);

            dispatch[kind].configure_repeats(device, frame,
                std::array<int, 8>{1, 1, 1, 1, 1, 1, 1, 1});
            network[kind] = std::make_unique<phoenix_native::Network>(
                device, dispatch[kind].api(), frame, weights, candidate, color.resource.Get(),
                motion.resource.Get(), allocation, allow_gpu_upload);
        }
    }
    void upload(ID3D12GraphicsCommandList* cmd) {
        for (auto* t : {&color, &motion}) {
            D3D12_TEXTURE_COPY_LOCATION src{t->transfer.Get(),
                                            D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, t->footprint};
            D3D12_TEXTURE_COPY_LOCATION dst{t->resource.Get(),
                                            D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0};
            cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            phoenix_native::transition(cmd, t->resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
    }
    void copy(ID3D12GraphicsCommandList* cmd, ID3D12Resource* image, unsigned b) {
        phoenix_native::transition(cmd, image, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION src{image, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0};
        D3D12_TEXTURE_COPY_LOCATION dst{readback[b].transfer.Get(),
                                        D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                                        readback[b].footprint};
        cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        phoenix_native::transition(cmd, image, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    std::vector<uint8_t> pixels(unsigned b) {
        auto& t = readback[b];
        void* map = nullptr;
        D3D12_RANGE range{0, SIZE_T(t.bytes)}, empty{};
        phoenix_native::hr(t.transfer->Map(0, &range, &map));
        std::vector<uint8_t> pixels(size_t(t.width) * t.height * 8);
        for (unsigned y = 0; y < t.height; ++y)
            memcpy(pixels.data() + size_t(y) * t.width * 8,
                   static_cast<uint8_t*>(map) + t.footprint.Offset +
                       y * t.footprint.Footprint.RowPitch,
                   size_t(t.width) * 8);
        t.transfer->Unmap(0, &empty);
        return pixels;
    }
};
// Submitted work must retire before payload destruction, also on failure.
struct Drain {
    phoenix_device::Owner& owner;
    ~Drain() {
        if (!owner.device() || FAILED(owner.device()->GetDeviceRemovedReason()))
            return;
        ComPtr<ID3D12Fence> fence;
        if (FAILED(owner.device()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))) ||
            FAILED(owner.queue()->Signal(fence.Get(), 1))) {
            TerminateProcess(GetCurrentProcess(), 75);
            return;
        }
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event || FAILED(fence->SetEventOnCompletion(1, event)) ||
            WaitForSingleObject(event, 30000) != WAIT_OBJECT_0)
            TerminateProcess(GetCurrentProcess(), 75);
        if (event)
            CloseHandle(event);
    }
};
int wmain(int argc, wchar_t** argv) {
    fs::path out;
    Json report{{"passed", false}, {"frames", Json::array()}};
    try {
        check(
            argc >= 5,
            "Usage: native_inference_probe BUNDLE CAPTURE_PREFIX "
            "NEW_OUTPUT WEIGHT_SOURCE_JSON [--cpu-only|--profile-layers|--check-upload-retirement|--staged-weights] [--streamed-weights] [--frames=N] [--warmup=N]");
        Json source_config;
        std::ifstream(argv[4]) >> source_config;
        auto source = phoenix_model::WeightSource::config(source_config);
        std::wstring option;
        unsigned frame_count = 128, warmup_frames = 16;
        bool streamed_weights = false;
        for (int argument = 5; argument < argc; ++argument) {
            const std::wstring value = argv[argument];
            const auto parse_count = [&](size_t prefix_length) {
                size_t consumed = 0;
                const auto count = std::stoul(value.substr(prefix_length), &consumed);
                check(consumed == value.size() - prefix_length && count <= 4096,
                      "Probe frame count must be between 0 and 4096");
                return static_cast<unsigned>(count);
            };
            if (value.rfind(L"--frames=", 0) == 0)
                frame_count = parse_count(9);
            else if (value.rfind(L"--warmup=", 0) == 0)
                warmup_frames = parse_count(9);
            else if (value == L"--streamed-weights")
                streamed_weights = true;
            else {
                check(option.empty(), "Only one probe execution mode may be selected");
                option = value;
            }
        }
        check(frame_count && warmup_frames < frame_count,
              "Probe warmup must leave at least one measured frame");
        const bool cpu_only = option == L"--cpu-only";
        const bool profile_layers = option == L"--profile-layers";
        const bool check_upload_retirement = option == L"--check-upload-retirement";
        const bool staged_weights = option == L"--staged-weights" || check_upload_retirement;
        check(option.empty() || cpu_only || profile_layers || staged_weights,
              "Unknown probe option");
        source.validate();
        out = argv[3];
        check(!fs::exists(out), "fresh output required");
        fs::create_directories(out);
        report["memory_phases"] = Json::array();
        report["memory_phases"].push_back(
            {{"stage", "before_model"}, {"memory", measure_memory(nullptr)}});
        const auto startup_start = std::chrono::steady_clock::now();
        const auto load_start = std::chrono::steady_clock::now();
        phoenix_bundle::Bundle bundle(
            argv[1], phoenix_model::mapped_file_digest, source, streamed_weights);
        std::vector<uint8_t> weight_storage;
        if (!streamed_weights) weight_storage = bundle.load_weights();
        auto weights = streamed_weights ? bundle.load_weight_view() : phoenix_model::WeightView(weight_storage);
        report["memory_phases"].push_back(
            {{"stage", "after_model_loaded"}, {"memory", measure_memory(nullptr)}});
        report["model_load_ms"] =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start)
                .count();
        report["weight_source"] = {{"kind", source.dll.empty() ? "files" : "nvidia_dll"},
                                   {"dll", source.dll.u8string()},
                                   {"dll_sha256", source.dll_sha256},
                                   {"weights_sha256", streamed_weights ? bundle.weights.at("sha256").get<std::string>() : digest(weight_storage)},
                                   {"streamed", streamed_weights},
                                   {"bytes", weights.size()}};
        report["comparison_kind"] = "single_instance_smoke";
        auto candidate = phoenix_kernel::load_catalog(phoenix_native::catalog(bundle));
        report["memory_phases"].push_back(
            {{"stage", "after_kernel_bytes_loaded"}, {"memory", measure_memory(nullptr)}});
        size_t kernel_capacity = 0;
        for (const auto& binary : candidate)
            kernel_capacity += binary.bytes.capacity();
        report["host_buffer_capacities"] = {{"model", weight_storage.capacity()},
                                            {"kernel_bytes", kernel_capacity}};
        const fs::path capture = argv[2];
        const auto meta_bytes = read(capture.string() + ".meta");
        phoenix::direct::CaptureMeta meta;
        check(phoenix::direct::capture_meta_parse(std::string(meta_bytes.begin(), meta_bytes.end()),
                                                  meta),
              "capture metadata");
        const Json* selected = &bundle.select_frame(meta.width, meta.height, [&](const Json& frame) {
            phoenix_native::preflight(frame, candidate);
        });
        check(meta.motion_bpp == 4, "supported motion format");
        const auto& frame = *selected;
        report["kernel_selections"] = Json::array();
        for (const auto& binary : candidate) {
            const auto& spec = binary.spec;
            const auto& entry = phoenix_kernel::entry_for_resolution(
                spec.entry, spec.large_input_entry, spec.large_input_min_pixels,
                meta.width, meta.height);
            report["kernel_selections"].push_back({{"name", spec.name}, {"entry", entry}});
        }
        report["generated_workspace_bytes"] = frame.at("arenas").at("workspace").at("bytes");
        report["generated_graph_sha256"] = bundle.manifest.at("native_graph_sha256");
        if (cpu_only) {
            report["passed"] = true;
            report["gpu_executed"] = false;
            report["geometry_exact"] = true;
            std::ofstream(out / "generated_frame.json") << frame.dump(2);
            std::ofstream(out / "result.json") << report.dump(2);
            return 0;
        }
        auto input = read(capture.string() + ".proxy.bin"),
             vectors = read(capture.string() + ".motion.bin");
        report["host_buffer_capacities"]["probe_input"] = input.capacity();
        report["host_buffer_capacities"]["probe_motion"] = vectors.capacity();
        report["input_sha256"] = digest(input);
        report["motion_sha256"] = digest(vectors);
        report["width"] = meta.width;
        report["height"] = meta.height;
        report["per_dispatch_queries"] = profile_layers;
        report["warmup_frames"] = warmup_frames;
        report["measured_frames"] = frame_count - warmup_frames;
        ComPtr<ID3D12Debug> debug;
        phoenix_native::hr(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
        debug->EnableDebugLayer();
        ComPtr<IDXGIFactory6> factory;
        phoenix_native::hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        LUID luid{};
        bool found = false;
        ComPtr<IDXGIAdapter1> measured_adapter;
        for (unsigned i = 0;; ++i) {
            ComPtr<IDXGIAdapter1> adapter;
            auto status = factory->EnumAdapterByGpuPreference(
                i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter));
            if (status == DXGI_ERROR_NOT_FOUND)
                break;
            phoenix_native::hr(status);
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            if (desc.VendorId == 0x10de && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                luid = desc.AdapterLuid;
                measured_adapter = adapter;
                found = true;
                break;
            }
        }
        check(found, "NVIDIA adapter required");
        report["memory_phases"].push_back({{"stage", "adapter_before_runtime"},
                                           {"memory", measure_memory(measured_adapter.Get())}});
        phoenix_device::Owner owner;
        owner.initialize(luid, {meta.width, meta.height},
                         [&](ID3D12Device* d, phoenix_device::Extent, uint64_t) {
                             D3D12_FEATURE_DATA_D3D12_OPTIONS16 options{};
                             report["gpu_upload_heap_supported"] =
                                 SUCCEEDED(d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16,
                                                                  &options, sizeof(options))) &&
                                 options.GPUUploadHeapSupported;
                             report["memory_phases"].push_back(
                                 {{"stage", "device_created_before_payload"},
                                  {"memory", measure_memory(measured_adapter.Get())}});
                             return std::make_unique<Payload>(d, frame, weights, candidate, input,
                                                              vectors, !staged_weights);
                         });
        report["initialization_ms"] = std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - startup_start)
                                          .count();
        report["memory_phases"].push_back(
            {{"stage", "runtime_created"}, {"memory", measure_memory(measured_adapter.Get())}});
        Drain drain{owner};
        auto* p = static_cast<Payload*>(owner.payload());
        report["host_buffer_capacities"]["network_containers"] =
            p->network[0]->host_container_capacity_bytes();
        report["owned_memory_samples"] = Json::array();
        report["owned_memory_samples"].push_back({{"stage", "before_upload"},
                                                  {"owned", p->memory.snapshot()},
                                                  {"probe_only", p->harness_memory.snapshot()}});
        std::unique_ptr<KernelTiming> layers;
        if (profile_layers) {
            layers = std::make_unique<KernelTiming>(owner.device(), owner.queue());
            kernel_timer = layers.get();
            untimed_dispatch = p->dispatch[0].api().dispatch;
            p->dispatch[0].api().dispatch = timed_kernel_dispatch;
        }
        phoenix::GpuTiming timer(owner.device(), owner.queue());
        active_timer = &timer;
        phoenix_native::group_marker = [](ID3D12GraphicsCommandList* cmd, unsigned group) {
            active_timer->group(cmd, group);
        };
        Controls controls;
        controls.enabled = meta.enabled;
        controls.auto_mask = meta.automask;
        controls.motion_x = meta.motion_x;
        controls.motion_y = meta.motion_y;
        controls.tone = meta.tone;
        controls.structure = meta.structure;
        controls.skin = meta.skin;
        controls.intensity = meta.intensity;
        if (check_upload_retirement) {
            phoenix_native::group_marker = nullptr;
            // Hold the queue before upload, submit two frames, and prove that
            // recording another frame cannot free an upload still in flight.
            // Always unblock on scope exit so failure cleanup can drain safely.
            struct QueueGate {
                ComPtr<ID3D12Fence> fence;
                ~QueueGate() {
                    if (fence)
                        fence->Signal(1);
                }
            } gate;
            phoenix_native::hr(
                owner.device()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate.fence)));
            phoenix_native::hr(owner.queue()->Wait(gate.fence.Get(), 1));
            phoenix_device::Ticket last;
            for (unsigned queued = 0; queued < 2; ++queued) {
                phoenix_device::Frame work;
                check(owner.try_begin(work) == phoenix_device::Status::complete,
                      "queued upload slot");
                if (!queued)
                    p->upload(work.commands);
                controls.reset = queued == 0;
                const auto resolved =
                    resolve_controls(controls, meta.width, meta.height, true, queued > 0,
                                     p->network[0]->next_sampling_index());
                FrameDispatch::Scope scope(p->dispatch[0], work.commands, resolved, 8);
                p->network[0]->record(work.commands, !queued, queued, owner.queue());
                scope.finish();
                last = owner.submit(work);
                p->network[0]->commit(work.commands);
            }
            const auto blocked = p->memory.snapshot();
            check(blocked["heap_allocations"]["upload"]["live_allocation_info_bytes"] > 0,
                  "in-flight upload was retired early");
            report["upload_retirement_check"]["blocked"] = blocked;
            phoenix_native::hr(gate.fence->Signal(1));
            check(owner.wait(last, 30000) == phoenix_device::Status::complete,
                  "queued upload fence");
            report["upload_retirement_check"]["passed"] = true;
            phoenix_native::group_marker = [](ID3D12GraphicsCommandList* commands, unsigned group) {
                active_timer->group(commands, group);
            };
        }
        const auto started = GetTickCount64();
        for (unsigned i = 0; i < frame_count; ++i) {
            check(GetTickCount64() - started < 300000, "profile wall limit");
            if (i == warmup_frames)
                timer.restart();
            phoenix_device::Frame work;
            check(owner.try_begin(work) == phoenix_device::Status::complete, "frame slot");
            if (!i && !check_upload_retirement)
                p->upload(work.commands);
            ID3D12Resource* outputs[2]{};
            double cpu_ms[2]{};
            controls.reset = i == 0;
            const auto host_started_unix_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();
            const auto call_start = std::chrono::steady_clock::now();
            if (layers)
                layers->begin_frame();
            for (unsigned order = 0; order < 1; ++order) {
                const unsigned b = 0;
                timer.begin(work.commands, b, 8, meta.width, meta.height);
                timer.own_begin();
                const auto cpu_start = std::chrono::steady_clock::now();
                const auto resolved =
                    resolve_controls(controls, meta.width, meta.height, true, i > 0,
                                     i == 0 ? 0 : p->network[b]->next_sampling_index());
                FrameDispatch::Scope scope(p->dispatch[b], work.commands, resolved, 8);
                outputs[b] = p->network[b]->record(work.commands, i == 0, i, owner.queue());
                scope.finish();
                cpu_ms[b] = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - cpu_start)
                                .count();
                timer.own_end();
                timer.end();
            }

            if (layers)
                layers->resolve(work.commands);
            ID3D12CommandList* lists[]{work.commands};
            timer.validate_submission(owner.queue(), 1, lists);
            auto ticket = owner.submit(work);
            p->network[0]->commit(work.commands);
            timer.submitted(owner.queue(), 1, lists);
            check(owner.wait(ticket, 30000) == phoenix_device::Status::complete, "profile fence");
            timer.poll();
            const auto retirement_start = GetTickCount64();
            while (!timer.idle()) {
                check(GetTickCount64() - retirement_start < 30000, "timing fence retirement");
                timer.poll();
                Sleep(0);
            }
            const double completed_ms = std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - call_start)
                                            .count();
            report["frames"].push_back(
                {{"frame", i},
                 {"warmup", i < warmup_frames},
                 // Correlate only the timed loop with external clock samples;
                 // model loading/upload also boost the GPU during process startup.
                 {"host_started_unix_ms", host_started_unix_ms},
                 {"host_completed_unix_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                                                std::chrono::system_clock::now().time_since_epoch())
                                                .count()},
                 {"cpu_record_ms", cpu_ms[0]},
                 {"record_submit_wait_ms", completed_ms},
                 {"gpu", timer.report()},
                 {"memory", measure_memory(measured_adapter.Get())}});
            if (i == 0 || i == 1 || i == warmup_frames || i == frame_count - 1)
                report["owned_memory_samples"].push_back(
                    {{"frame", i},
                     {"owned", p->memory.snapshot()},
                     {"probe_only", p->harness_memory.snapshot()}});
            if (layers)
                report["frames"].back()["kernel_ms"] = layers->read();
            check(!debug_errors(owner.device()), "D3D12 debug error");
            if (i == 0 || i == frame_count - 1) {
                phoenix_device::Frame copy;
                check(owner.try_begin(copy) == phoenix_device::Status::complete, "readback slot");
                p->copy(copy.commands, outputs[0], 0);
                auto copied = owner.submit(copy);
                check(owner.wait(copied, 30000) == phoenix_device::Status::complete,
                      "readback fence");
                auto pixels = p->pixels(0);
                bool finite = true;
                for (size_t k = 0; k < pixels.size(); k += 2) {
                    uint16_t value = 0;
                    memcpy(&value, pixels.data() + k, 2);
                    finite = finite && ((value & 0x7c00) != 0x7c00);
                }
                check(finite, "Nonfinite native output");
                report["output_finite"] = finite;
                report["output_sha256"] = digest(pixels);
                report["output_hashes"][std::to_string(i)] = digest(pixels);
            }
        }
        phoenix_native::group_marker = nullptr;
        active_timer = nullptr;
        report["memory_snapshot"] = p->memory.snapshot();
        check(
            report["memory_snapshot"]["heap_allocations"]["upload"]["live_allocation_info_bytes"] ==
                0,
            "completed model upload remains allocated");
        report["probe_only_memory_snapshot"] = p->harness_memory.snapshot();
        auto owned_ledger = p->memory;
        auto harness_ledger = p->harness_memory;
        const auto c = p->network[0]->counters;
        check(owner.shutdown(30000) == phoenix_device::Status::complete, "profile shutdown");
        check(c->modules_destroyed == 41 && !c->destroy_failures, "module retirement");
        report["owned_memory_after_shutdown"] = owned_ledger.snapshot();
        report["probe_only_memory_after_shutdown"] = harness_ledger.snapshot();
        report["memory_phases"].push_back(
            {{"stage", "runtime_released"}, {"memory", measure_memory(measured_adapter.Get())}});
        // This probe retains its host inputs through timing, then measures
        // their release separately. Production readers end after GPU upload.
        weights = {};
        std::vector<uint8_t>().swap(weight_storage);
        std::vector<uint8_t>().swap(input);
        std::vector<uint8_t>().swap(vectors);
        std::vector<phoenix_kernel::Binary>().swap(candidate);
        report["memory_phases"].push_back({{"stage", "host_buffers_released"},
                                           {"memory", measure_memory(measured_adapter.Get())}});
        report["scope"] =
            "Single optimized native RTX instance, provided normalized FP16 input/motion; GPU "
            "whole runner interval, CPU record and record/submit/fence wall separated; input "
            "uploads and output readback excluded; model tensors never written; smoke success "
            "alone does not establish reference equivalence";
        report["gpu_executed"] = true;
        report["passed"] = true;
        report["modules_retired"] = 41;
        report["debug_errors"] = 0;
        std::ofstream(out / "result.json") << report.dump(2);
        std::cout << frame_count << " single-instance frames, " << frame_count - warmup_frames
                  << " measured, zero debug errors\n";
        return 0;
    } catch (const std::exception& e) {
        report["error"] = e.what();
        if (!out.empty() && fs::exists(out))
            std::ofstream(out / "result.json") << report.dump(2);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
