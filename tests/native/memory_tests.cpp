#include "../../src/native/support/memory.h"
#include "../../src/native/support/host_memory.h"
#include <iostream>
#include <thread>
using namespace phoenix_native::memory;
static unsigned checks = 0;
static void expect(bool value) {
    ++checks;
    if (!value)
        throw std::runtime_error("allocation accounting test failed at check " +
                                 std::to_string(checks));
}
class Object final : public ID3D12Object {
    std::atomic<ULONG> refs_{1};
    Microsoft::WRL::ComPtr<IUnknown> token_;

  public:
    bool refuse = false;
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override {
        if (!out)
            return E_POINTER;
        *out = nullptr;
        if (iid != __uuidof(IUnknown) && iid != __uuidof(ID3D12Object))
            return E_NOINTERFACE;
        *out = static_cast<ID3D12Object*>(this);
        AddRef();
        return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override {
        return ++refs_;
    }
    ULONG STDMETHODCALLTYPE Release() override {
        const auto count = --refs_;
        if (!count)
            delete this;
        return count;
    }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override {
        return E_NOTIMPL;
    }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override {
        return E_NOTIMPL;
    }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID guid, const IUnknown* data) override {
        if (refuse || guid != release_cookie)
            return E_FAIL;
        token_ = const_cast<IUnknown*>(data);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override {
        return S_OK;
    }
};
int main() {
    try {
        namespace hm = phoenix_addon::host_memory;
        auto& host = hm::counters()[size_t(hm::Tag::resources)];
        {
            hm::Map<unsigned, unsigned, hm::Tag::resources> entries;
            const auto empty_bytes = host.live.load();
            entries.emplace(1, 2);
            expect(host.live > sizeof(unsigned) * 2);
            entries.clear();
            expect(host.live == empty_bytes && host.allocated - host.freed == empty_bytes);
        }
        expect(host.live == 0 && host.allocated == host.freed);
        {
            hm::Vector<unsigned, hm::Tag::resources> values;
            values.reserve(16);
            values.push_back(3);
            const auto retained = host.live.load();
            values.clear();
            expect(host.live == retained);
            auto moved = std::move(values);
            expect(host.live == retained && moved.capacity() >= 16);
        }
        expect(host.live == 0);
        {
            auto value =
                std::allocate_shared<unsigned>(hm::Allocator<unsigned, hm::Tag::resources>{}, 7);
            std::weak_ptr<unsigned> weak = value;
            expect(host.live > 0);
            value.reset();
            expect(weak.expired() && host.live > 0);
            weak.reset();
            expect(host.live == 0);
        }
        std::vector<std::thread> host_threads;
        for (unsigned iteration_index = 0; iteration_index < 4; ++iteration_index)
            host_threads.emplace_back([] {
                for (unsigned iteration = 0; iteration < 1000; ++iteration) {
                    hm::Map<unsigned, unsigned, hm::Tag::resources> m;
                    m.emplace(iteration, iteration);
                }
            });
        for (auto& host_thread : host_threads)
            host_thread.join();
        expect(host.live == 0 && host.allocated == host.freed &&
               host.allocations == host.deallocations);
        Ledger ledger;
        auto generation = ledger.begin(640, 384);
        Microsoft::WRL::ComPtr<Object> object;
        object.Attach(new Object);
        track(object.Get(), generation, Kind::weights, 65536);
        auto lease = object;
        expect(ledger.snapshot()["totals"]["live_allocation_info_bytes"] == 65536);
        object.Reset();
        expect(generation->total.released == 0);
        {
            SessionLifetime lifetime{generation};
        }
        expect(generation->session_destroyed && !generation->reclaimable());
        lease.Reset();
        expect(generation->reclaimable());
        expect(generation->total.freed == 65536 && generation->total.released == 1);
        expect(ledger.snapshot()["generations"][0]["categories"]["weights"]["live_objects"] == 0);

        auto partial = ledger.begin(1280, 720);
        try {
            SessionLifetime lifetime{partial};
            Microsoft::WRL::ComPtr<Object> first;
            first.Attach(new Object);
            track(first.Get(), partial, Kind::color, 4096);
            throw std::runtime_error("construction failure");
        } catch (const std::runtime_error&) {
        }
        expect(!partial->constructed && partial->session_destroyed && partial->reclaimable());
        expect(partial->total.allocated == partial->total.freed);
        expect(ledger.snapshot()["totals"]["peak_allocation_info_bytes"] == 65536);

        auto failed = ledger.begin(640, 384);
        object.Attach(new Object);
        object->refuse = true;
        bool threw = false;
        try {
            track(object.Get(), failed, Kind::history, 8192);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        expect(threw && !failed->accounting_valid &&
               failed->total.created == failed->total.released);
        object.Reset();
        object.Attach(new Object);
        track(object.Get(), {}, Kind::fence);
        object.Reset();
        expect(ledger.snapshot()["generations_started"] == 3);
        object.Attach(new Object);
        track(object.Get(), failed, Kind::fence);
        expect(failed->total.created == 2 && failed->total.allocated == 8192);
        object.Reset();
        expect(failed->kinds[size_t(Kind::fence)].released == 1);

        Ledger bounded;
        auto held = bounded.begin(1, 1);
        object.Attach(new Object);
        track(object.Get(), held, Kind::workspace, 123);
        {
            SessionLifetime lifetime{held};
        }
        for (unsigned iteration_index = 0; iteration_index < 160; ++iteration_index) {
            auto generation = bounded.begin(2, 2);
            SessionLifetime lifetime{generation};
            Microsoft::WRL::ComPtr<Object> local;
            local.Attach(new Object);
            track(local.Get(), generation, Kind::upload, 10);
        }
        const auto snapshot = bounded.snapshot();
        expect(snapshot["generations_started"] == 161 &&
               snapshot["closed_records_compacted"] == 33);
        expect(snapshot["generations"].size() == 128 &&
               snapshot["generations"][0]["generation_id"] == 1);
        expect(snapshot["totals"]["allocation_info_bytes_created"] == 1723);
        expect(snapshot["totals"]["allocation_info_bytes_released"] == 1600);
        object.Reset();
        expect(bounded.snapshot()["totals"]["live_allocation_info_bytes"] == 0);

        Ledger concurrent;
        auto concurrent_generation = concurrent.begin(1, 1);
        std::vector<std::thread> threads;
        for (unsigned iteration_index = 0; iteration_index < 4; ++iteration_index)
            threads.emplace_back([concurrent_generation] {
                for (unsigned iteration = 0; iteration < 1000; ++iteration) {
                    auto* release_token =
                        new ReleaseToken(concurrent_generation, Kind::history, 64);
                    release_token->Release();
                }
            });
        for (auto& thread : threads)
            thread.join();
        expect(concurrent.snapshot()["totals"]["objects_created"] == 4000);
        expect(concurrent.snapshot()["totals"]["object_release_callbacks"] == 4000);
        expect(concurrent.snapshot()["totals"]["live_allocation_info_bytes"] == 0);
        concurrent_generation->modules = std::make_shared<phoenix_kernel::Counters>();
        concurrent_generation->modules->modules_created = 1;
        concurrent_generation->session_destroyed = true;
        expect(!concurrent_generation->reclaimable());
        concurrent_generation->modules->modules_destroyed = 1;
        expect(concurrent_generation->reclaimable());
        concurrent_generation->modules->destroy_failures = 1;
        expect(!concurrent_generation->reclaimable());
        auto* token = new ReleaseToken(concurrent_generation, Kind::views, 0);
        void* queried = nullptr;
        expect(token->QueryInterface(__uuidof(IUnknown), nullptr) == E_POINTER);
        expect(token->QueryInterface(__uuidof(IUnknown), &queried) == S_OK);
        expect(token->Release() == 1);
        expect(static_cast<IUnknown*>(queried)->Release() == 0);
        Ledger heap_ledger;
        auto heap_generation = heap_ledger.begin(1, 1);
        object.Attach(new Object);
        track(object.Get(), heap_generation, Kind::upload, 65536, Heap::upload);
        auto heap_snapshot = heap_ledger.snapshot();
        expect(heap_snapshot["heap_allocations"]["upload"]["live_allocation_info_bytes"] == 65536);
        expect(heap_snapshot["heap_allocations"]["device_default"]["live_allocation_info_bytes"] ==
               0);
        expect(heap_generation->json()["categories"]["upload"]["live_allocation_info_bytes"] ==
               65536);
        object.Reset();
        heap_snapshot = heap_ledger.snapshot();
        expect(heap_snapshot["heap_allocations"]["upload"]["live_allocation_info_bytes"] == 0);
        expect(heap_snapshot["heap_allocations"]["upload"]["allocation_info_bytes_released"] ==
               65536);
        expect(heap_snapshot["heap_allocations"]["upload"]["peak_allocation_info_bytes"] == 65536);
        std::cout
            << Json{{"passed", true},
                    {"checks", checks},
                    {"gpu_executed", false},
                    {"scope",
                     "Real accounting classes and COM private-interface lifetime mock; resource allocation-size query requires live GPU gate"}}
                   .dump(2)
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
