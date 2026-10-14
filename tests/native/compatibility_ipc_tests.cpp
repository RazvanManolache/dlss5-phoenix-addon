// CPU-only failure/ownership tests. No graphics device or game is created.
#include "../../src/integration/compatibility/host_client.h"
#include "../../src/integration/compatibility/host/retired_transfer_buffer.h"
#include "../../src/integration/compatibility/host/phoenix-host-safety.h"
#include <iostream>

struct FakeResource {
    unsigned& released;
    void Release() {
        ++released;
    }
};
int main() try {
    unsigned checks = 0;
    auto check = [&](bool value) {
        ++checks;
        if (!value)
            throw std::runtime_error("IPC lifecycle contract failed at check " +
                                     std::to_string(checks));
    };
    using namespace phoenix_compat;
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
    const auto cleanup_parent = fs::canonical(fs::current_path());
    const std::string cleanup_name = "ipc-owned-session-" + std::to_string(GetCurrentProcessId());
    const auto cleanup_folder = cleanup_parent / cleanup_name;
    fs::create_directory(cleanup_folder);
    std::ofstream(cleanup_folder / "ReShade.ini") << "temporary test configuration";
    cleanup_stopped_session(cleanup_folder, cleanup_parent, "different-session");
    check(fs::is_regular_file(cleanup_folder / "ReShade.ini"));
    std::ofstream(cleanup_folder / "capture.rgba") << "preserved capture";
    cleanup_stopped_session(cleanup_folder, cleanup_parent, cleanup_name);
    check(fs::is_regular_file(cleanup_folder / "capture.rgba"));
    fs::remove(cleanup_folder / "capture.rgba");
    cleanup_stopped_session(cleanup_folder, cleanup_parent, cleanup_name);
    check(!fs::exists(cleanup_folder));
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
