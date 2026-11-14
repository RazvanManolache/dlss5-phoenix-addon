// CPU-only failure/ownership tests. No graphics device or game is created.
#include "../../src/integration/compatibility/host_client.h"
#include "../../src/integration/compatibility/pixel_transport.h"
#include "../../src/integration/compatibility/host/retired_transfer_buffer.h"
#include "../../src/integration/compatibility/host/phoenix-host-safety.h"
#include "../../src/integration/compatibility/dx_depth_readback.h"
#include <iostream>
#include <source_location>

struct FakeResource {
    unsigned& released;
    void Release() {
        ++released;
    }
};
static HRESULT WINAPI unused_evaluate(const PhoenixFeederFrameV1*, PhoenixFeederResultV1*) { return E_NOTIMPL; }
static UINT WINAPI unused_metadata(char*, UINT) { return 0; }
static BOOL WINAPI unused_action(uint32_t, uint32_t) { return FALSE; }
static phoenix_compat::InProcessApi provider_callbacks() {
    phoenix_compat::InProcessApi api;
    api.prepare = [](auto*, const auto&) {};
    api.release = [](auto*) {};
    api.observe = [](auto*, auto*, auto*) {};
    api.forget = [](ID3D12GraphicsCommandList*) noexcept {};
    api.evaluate = unused_evaluate; api.progress = api.ui = unused_metadata;
    api.action = unused_action; api.update_tuning = [](const auto&) {};
    return api;
}
int main() try {
    unsigned checks = 0;
    auto check = [&](bool value, std::source_location location = std::source_location::current()) {
        ++checks;
        if (!value)
            throw std::runtime_error("IPC lifecycle contract failed at check " +
                                     std::to_string(checks) + " (line " + std::to_string(location.line()) + ")");
    };
    using namespace phoenix_compat;
    {
        const uint8_t depth16[]{0x00, 0x80};
        const uint8_t depth24_stencil8[]{0x00, 0x00, 0x80, 0xff};
        const uint8_t depth32_stencil8[]{0x00, 0x00, 0x80, 0x3e, 0xff, 0xff, 0xff, 0xff};
        check(depth_pixel_bytes(DXGI_FORMAT_D16_UNORM) == sizeof(depth16));
        check(depth_pixel_bytes(DXGI_FORMAT_D24_UNORM_S8_UINT) == sizeof(depth24_stencil8));
        check(depth_pixel_bytes(DXGI_FORMAT_D32_FLOAT_S8X24_UINT) == sizeof(depth32_stencil8));
        check(depth_staging_format(DXGI_FORMAT_D16_UNORM) == DXGI_FORMAT_R16_TYPELESS);
        check(depth_staging_format(DXGI_FORMAT_D24_UNORM_S8_UINT) == DXGI_FORMAT_R24G8_TYPELESS);
        check(depth_staging_format(DXGI_FORMAT_D32_FLOAT) == DXGI_FORMAT_R32_TYPELESS);
        check(depth_staging_format(DXGI_FORMAT_D32_FLOAT_S8X24_UINT) == DXGI_FORMAT_R32G8X24_TYPELESS);
        check(std::abs(unpack_device_depth(depth16, DXGI_FORMAT_D16_UNORM) - 0.5f) < 0.00001f);
        check(std::abs(unpack_device_depth(depth24_stencil8, DXGI_FORMAT_D24_UNORM_S8_UINT) - 0.5f) < 0.00001f);
        check(unpack_device_depth(depth32_stencil8, DXGI_FORMAT_D32_FLOAT_S8X24_UINT) == 0.25f);
    }
    {
        // Force detach while a worker still owns callbacks. The provider may
        // neither disappear nor be replaced until its final cleanup callback.
        Handle entered, resume;
        entered.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        resume.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        check(entered.value && resume.value);
        std::atomic<unsigned> cleaned = 0;
        auto provider = std::make_shared<int>(1);
        std::weak_ptr<int> provider_weak = provider;
        auto api = provider_callbacks();
        api.release = [provider, &cleaned](auto*) { ++cleaned; };
        check(publish_in_process_api(std::move(api)));
        auto owner = lease_in_process_api();
        std::jthread worker([&, callbacks = lease_in_process_api()] {
            SetEvent(entered.value);
            WaitForSingleObject(resume.value, 2000);
            in_process_api.release(nullptr);
        });
        struct Resume { HANDLE event; ~Resume() { SetEvent(event); } } unblock{resume.value};
        check(WaitForSingleObject(entered.value, 1000) == WAIT_OBJECT_0);
        provider.reset();
        retire_in_process_api();
        check(!in_process_api_available() && !can_publish_in_process_api());
        check(!publish_in_process_api(provider_callbacks()));
        bool denied = false;
        try { (void)lease_in_process_api(); } catch (const std::exception&) { denied = true; }
        check(denied && !provider_weak.expired() && cleaned == 0);
        SetEvent(resume.value); worker.join();
        check(cleaned == 1 && !provider_weak.expired());
        owner.reset();
        check(provider_weak.expired() && !in_process_api.release && can_publish_in_process_api());
        retire_in_process_api(); // idempotent, including the no-owner case
        check(publish_in_process_api(provider_callbacks()) && in_process_api_available());
        retire_in_process_api();
        check(!in_process_api_available() && !in_process_api.evaluate);
    }
    check(!feed_protocol_compatible(9));
    check(feed_protocol_compatible(10));
    check(feed_protocol_compatible(11));
    check(feed_protocol_compatible(12));
    check(feed_protocol_compatible(13));
    check(feed_protocol_compatible(14));
    check(feed_protocol_compatible(15));
    check(feed_protocol_compatible(16) && !feed_protocol_compatible(17));
    check(feed_frame_msg_bytes(14) == 24 && feed_frame_msg_bytes(15) == sizeof(FeedFrameMsg));
    check(feed_cpu_depth_bytes(14) == 8 && feed_cpu_mapping_bytes(14) == 8);
    {
        CpuFrameMapping producer, consumer;
        check(!producer.create(0, 1080) && !producer.create(8192, 8192));
        check(producer.create(256, 128));
        check(!consumer.open(GetCurrentProcess(), producer.handle(), 2560, 1080));
        check(consumer.open(GetCurrentProcess(), producer.handle(), 256, 128));
        producer.input().front() = 31;
        producer.depth().back() = .375f;
        consumer.output().back() = 209;
        check(consumer.input().front() == 31 && consumer.depth().back() == .375f);
        check(producer.output().back() == 209 && producer.output().front() == 0);
        producer.reset();
        check(consumer.input().front() == 31);
    }
    {
        // Ring slots are isolated planes in one mapping, visible to the peer.
        CpuFrameMapping producer, consumer;
        check(!producer.create(64, 32, CpuFrameMapping::max_slots() + 1));
        check(producer.create(64, 32, 3) && producer.slots() == 3);
        check(consumer.open(GetCurrentProcess(), producer.handle(), 64, 32, 3));
        producer.input(2).front() = 77;
        producer.depth(1).back() = .25f;
        consumer.output(0).front() = 55;
        check(consumer.input(2).front() == 77 && consumer.input(0).front() == 0);
        check(consumer.depth(1).back() == .25f && consumer.depth(2).back() == 0.f);
        check(producer.output(0).front() == 55 && producer.output(2).front() == 0);
        check(producer.input(3).empty() && producer.output(9).empty());
    }
    {
        CpuFrameMapping producer, consumer;
        check(producer.create(257, 255, 3, true));
        check(consumer.open(GetCurrentProcess(), producer.handle(), 257, 255, 3, true));
        producer.input(1).front() = 121;
        producer.depth(1).front() = .42f;
        producer.publish(1, 4294967297ull, true, feed_pack_depth_dims(128, 127));
        check(consumer.ready_frame(0) == 0 && consumer.ready_frame(1) == 4294967297ull);
        check(consumer.input(1).front() == 121 && consumer.depth(1).front() == .42f);
        check(consumer.depth_ready(1) && consumer.depth_dims(1) == feed_pack_depth_dims(128, 127));
        producer.publish(1, 4294967298ull, false, 0);
        check(consumer.ready_frame(1) == 4294967298ull && !consumer.depth_ready(1));
        check(feed_cpu_mapping_bytes(15) == 16 && feed_cpu_mapping_bytes(16) == sizeof(FeedCpuMapping));
    }
    {
        // Adaptive ring depth: starts at two, grows only on starvation,
        // shrinks only on backlog, and never leaves [2, slots].
        CpuRingDepth depth(4);
        check(depth.limit() == 2 && depth.admits(1) && !depth.admits(2));
        depth.starved();
        check(depth.limit() == 3);
        depth.starved();
        depth.starved();
        check(depth.limit() == 4);
        depth.backlogged();
        check(depth.limit() == 3);
        depth.backlogged();
        depth.backlogged();
        check(depth.limit() == 2);
        CpuRingDepth single(1);
        single.starved();
        check(single.limit() == 1 && single.admits(0) && !single.admits(1));
        CpuRingSlots slots;
        const auto first = slots.acquire(4);
        const auto abandoned = slots.acquire(4);
        const auto last = slots.acquire(4);
        slots.release(abandoned);
        for (unsigned i = 0; i < 12; ++i) {
            const auto next = slots.acquire(4);
            check(next == abandoned && next != first && next != last);
            slots.release(next);
        }
        check(slots.size() == 2);
    }
    {
        // BGRA swap (SIMD where available) matches the scalar channel swap,
        // including a width that leaves a scalar tail.
        std::vector<uint8_t> source(7 * 2 * 4), swapped(7 * 2 * 4), expected(7 * 2 * 4);
        for (size_t i = 0; i < source.size(); ++i) source[i] = uint8_t(i * 37 + 5);
        for (size_t pixel = 0; pixel < expected.size() / 4; ++pixel) {
            expected[pixel * 4] = source[pixel * 4 + 2];
            expected[pixel * 4 + 1] = source[pixel * 4 + 1];
            expected[pixel * 4 + 2] = source[pixel * 4];
            expected[pixel * 4 + 3] = source[pixel * 4 + 3];
        }
        copy_pixel_rows(swapped.data(), 7 * 4, source.data(), 7 * 4, 7, 2, false, true);
        check(swapped == expected);
        std::vector<uint8_t> plain(16);
        copy_pixel_rows(plain.data(), 8, source.data(), 8, 2, 2, false, false);
        check(std::equal(plain.begin(), plain.end(), source.begin()));
    }
    FeedGuideState pending{1,1920.f,-1080.f};
    check(feed_guides_valid(pending));
    const auto declared=feed_take_guides(pending);
    check(declared.engine_motion==1&&declared.scale_x==1920.f&&declared.scale_y==-1080.f);
    check(!feed_take_guides(pending).engine_motion);
    check(!feed_guides_valid({2,1,1}));
    check(!feed_guides_valid({1,std::numeric_limits<float>::infinity(),1}));
    {
        Settings tuning;
        auto state = std::make_shared<HostState>(tuning); // No start: no helper, GPU or pipe.
        check(state->tuning_current_output());
        check(state->update_tuning(tuning));
        check(state->ui_snapshot()["live_tuning"]["requested_revision"] == 0);
        tuning.controls.enabled = true;
        for (float tone : {.3f, .6f, 1.2f}) {
            tuning.controls.tone = tone;
            check(state->update_tuning(tuning));
        }
        auto snapshot = state->ui_snapshot()["live_tuning"];
        check(snapshot["requested_revision"] == 3 && snapshot["applied_revision"] == 0 && snapshot["pending"] == true);
        check(!state->tuning_current_output()); // Old frame cannot acknowledge new slider settings.
        check(state->update_tuning(tuning));
        check(state->ui_snapshot()["live_tuning"]["requested_revision"] == 3); // Coalesces unchanged frames.
        tuning.controls.skin = 100;
        check(!state->update_tuning(tuning));
        check(state->ui_snapshot()["live_tuning"]["requested_revision"] == 3);
        state->stop();
    }
    // A hung fence or failed queue signal must retain the resource and deny
    // resize/reuse. Later proven completion is the only release permission.
    unsigned allocations = 0, releases = 0;
    FakeResource first{releases}, second{releases};
    {
        PhoenixRetiredTransferBuffer<FakeResource> buffer;
        auto allocate = [&](uint64_t) { return ++allocations == 1 ? &first : &second; };
        check(buffer.ensure(1024, allocate));
        check(buffer.ensure(512, allocate) && allocations == 1);
        buffer.submitted();
        check(!buffer.ensure(2048, allocate) && allocations == 1 && releases == 0);
        check(!buffer.completed(false)); // timeout, device loss, or signal failure
        buffer.release();
        check(releases == 0);
        check(!buffer.ensure(1024, allocate));
        check(buffer.completed(true));
        check(buffer.ensure(2048, allocate) && allocations == 2 && releases == 1);
    }
    check(releases == 2);
    {
        PhoenixRetiredTransferBuffer<FakeResource> buffer;
        check(buffer.ensure(1024, [&](uint64_t) { return &first; }));
        buffer.submitted();
        buffer.completed(false);
    }
    check(releases == 2); // uncertain-resource destructor must not call Release

    // Exercise submission failure after Execute and delayed completion as two
    // distinct injected faults, using the helper's actual submission/wait logic.
    for (bool signal_failure : {false, true}) {
        unsigned fault_releases = 0, fault_allocations = 0;
        FakeResource resource{fault_releases};
        {
            PhoenixRetiredTransferBuffer<FakeResource> buffer;
            check(buffer.ensure(1024, [&](uint64_t) {
                ++fault_allocations;
                return &resource;
            }));
            bool executed = false;
            buffer.submitted();
            const bool signaled = PhoenixSubmitCommands(
                [] { return true; }, [&] { executed = true; }, [&] { return !signal_failure; });
            check(executed && signaled != signal_failure);
            const uint64_t target = signaled ? 1 : kPhoenixFailedSubmission;
            const bool retired = PhoenixWaitFence(
                target, 1, [] { return uint64_t(0); }, [] { return true; },
                [](uint32_t) { return false; }, [] { return uint64_t(0); });
            check(!buffer.completed(retired));
            buffer.release();
            check(!buffer.ensure(2048, [&](uint64_t) {
                ++fault_allocations;
                return &resource;
            }));
            check(fault_releases == 0 && fault_allocations == 1);
        }
        check(fault_releases == 0);
    }

    // The actual cache-lock waiter must notice cancellation independently of
    // synchronous I/O. Main owns the mutex while another thread attempts it.
    Handle mutex;
    mutex.value = CreateMutexW(nullptr, TRUE, nullptr);
    check(mutex.value != nullptr);
    std::atomic<bool> cancelled = false;
    DWORD waited = WAIT_OBJECT_0;
    std::thread waiter(
        [&] { waited = phoenix_embedded::wait_cache_lock(mutex.value, &cancelled); });
    Sleep(25);
    const auto cancellation_start = GetTickCount64();
    cancelled = true;
    waiter.join();
    const auto cancellation_ms = GetTickCount64() - cancellation_start;
    check(waited == WAIT_FAILED && cancellation_ms < 250);
    ReleaseMutex(mutex.value);
    check(phoenix_embedded::wait_cache_lock(mutex.value, nullptr, 1000) == WAIT_OBJECT_0);
    ReleaseMutex(mutex.value);
    bool hash_cancelled = false;
    try {
        phoenix_embedded::file_digest(L"unused-on-cancel", &cancelled);
    } catch (const std::exception&) {
        hash_cancelled = true;
    }
    check(hash_cancelled);

    // Distinct sessions in one process can own independent pipe endpoints.
    char pipe_a[128]{}, pipe_b[128]{};
    sprintf_s(pipe_a, FEED_SESSION_PIPE_FMT, GetCurrentProcessId(), 1ull);
    sprintf_s(pipe_b, FEED_SESSION_PIPE_FMT, GetCurrentProcessId(), 2ull);
    Handle endpoint_a, endpoint_b;
    endpoint_a.value = CreateNamedPipeA(pipe_a, PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
                                        PIPE_TYPE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1,
                                        256, 256, 0, nullptr);
    endpoint_b.value = CreateNamedPipeA(pipe_b, PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
                                        PIPE_TYPE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1,
                                        256, 256, 0, nullptr);
    check(endpoint_a.value != INVALID_HANDLE_VALUE && endpoint_b.value != INVALID_HANDLE_VALUE);

    // Fence ownership stays in the source process until the client makes its
    // own duplicates. A failed second duplicate must roll back the first one.
    Handle input, output;
    input.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    output.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    FeedBuildAck ack{1, 1, uint64_t(uintptr_t(input.value)), uint64_t(uintptr_t(output.value))};
    DWORD handles_before = 0, handles_after = 0;
    check(GetProcessHandleCount(GetCurrentProcess(), &handles_before));
    auto owned = duplicate_fence_ack(GetCurrentProcess(), ack);
    check(owned.fence_in != ack.fence_in && owned.fence_out != ack.fence_out);
    check(SetEvent(input.value) &&
          WaitForSingleObject(HANDLE(uintptr_t(owned.fence_in)), 0) == WAIT_OBJECT_0);
    CloseHandle(HANDLE(uintptr_t(owned.fence_in)));
    CloseHandle(HANDLE(uintptr_t(owned.fence_out)));
    ack.fence_out = 0x12345678u;
    bool rejected = false;
    try {
        (void)duplicate_fence_ack(GetCurrentProcess(), ack);
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected);
    check(GetProcessHandleCount(GetCurrentProcess(), &handles_after) &&
          handles_after == handles_before);
    check(SetEvent(input.value) && SetEvent(output.value)); // originals remain valid

    uint64_t revision = 0;
    FeedUiAction clear{uint32_t(phoenix_ui::Action::ClearTiming), 0};
    revision = acknowledged_clear_revision(revision, clear, 0);
    check(revision == 0); // queued/rejected is not acknowledgement
    revision = acknowledged_clear_revision(revision, {uint32_t(phoenix_ui::Action::Marker), 0}, 1);
    check(revision == 0);
    revision = acknowledged_clear_revision(revision, clear, 1);
    check(revision == 1);
    revision = acknowledged_clear_revision(revision, clear, 1);
    check(revision == 2);

    // Cleanup cannot escape its named session or remove preserved captures.
    const auto data_root = fs::canonical(fs::current_path()) / L"central-storage";
    check(SetEnvironmentVariableW(L"PHOENIX_DATA_ROOT", data_root.c_str()));
    const auto [runtime, cleanup_parent] = prepare_helper_directories();
    check(fs::equivalent(runtime, data_root / L"runtime"));
    check(cleanup_parent == data_root / L"sessions");
    check(fs::equivalent(phoenix_embedded::default_cache(), runtime));
    check(prepare_helper_directories().runtime == runtime);
    const std::string cleanup_name = "ipc-owned-session-" + std::to_string(GetCurrentProcessId());
    const auto cleanup_folder = cleanup_parent / cleanup_name;
    fs::create_directory(cleanup_folder);
    std::ofstream(cleanup_folder / "ReShade.ini") << "temporary test configuration";
    const auto runtime_file = runtime / "helper.bin";
    const auto session_link = cleanup_folder / "helper.exe";
    std::ofstream(runtime_file) << "shared runtime";
    check(CreateHardLinkW(session_link.c_str(), runtime_file.c_str(), nullptr));
    cleanup_stopped_session(cleanup_folder, cleanup_parent, "different-session");
    check(fs::is_regular_file(cleanup_folder / "ReShade.ini"));
    std::ofstream(cleanup_folder / "capture.rgba") << "preserved capture";
    cleanup_stopped_session(cleanup_folder, cleanup_parent, cleanup_name);
    check(fs::is_regular_file(cleanup_folder / "capture.rgba"));
    fs::remove(cleanup_folder / "capture.rgba");
    cleanup_stopped_session(cleanup_folder, cleanup_parent, cleanup_name);
    check(!fs::exists(cleanup_folder));
    check(fs::is_regular_file(runtime_file));
    // A missing directory or a file where a directory was expected can throw
    // during filesystem enumeration. Cleanup must remain safe and best effort.
    static_assert(noexcept(cleanup_stopped_session(cleanup_folder, cleanup_parent, cleanup_name)));
    cleanup_stopped_session(cleanup_folder, cleanup_parent, cleanup_name);
    check(!fs::exists(cleanup_folder));
    std::ofstream(cleanup_folder) << "not a session directory";
    cleanup_stopped_session(cleanup_folder, cleanup_parent, cleanup_name);
    check(fs::is_regular_file(cleanup_folder));
    fs::remove(cleanup_folder);

    // Exercise the exact shared-ownership launcher used by HostState. The
    // game-facing lease disappears promptly while the worker is still waiting.
    Handle started, finish;
    started.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    finish.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    std::atomic<unsigned> destroyed = 0;
    struct State {
        std::atomic<unsigned>& destroyed;
        ~State() {
            ++destroyed;
        }
    };
    auto lease = std::make_shared<State>(destroyed);
    std::weak_ptr<State> weak = lease;
    start_owned_worker(lease, [&](State&) {
        SetEvent(started.value);
        WaitForSingleObject(finish.value, 2000);
    });
    check(WaitForSingleObject(started.value, 1000) == WAIT_OBJECT_0);
    const auto release_start = GetTickCount64();
    lease.reset();
    const auto release_ms = GetTickCount64() - release_start;
    check(release_ms < 50 && !weak.expired() && destroyed == 0);
    SetEvent(finish.value);
    const auto until = GetTickCount64() + 1000;
    while ((!weak.expired() || destroyed != 1) && GetTickCount64() < until)
        Sleep(1);
    check(weak.expired() && destroyed == 1);
    std::cout << "{\"passed\":true,\"gpu_executed\":false,\"pointer_bits\":" << sizeof(void*) * 8
              << ",\"checks\":" << checks << ",\"cache_cancel_ms\":" << cancellation_ms
              << ",\"owned_lease_release_ms\":" << release_ms << "}\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
}
