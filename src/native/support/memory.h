#pragma once
#include "../runtime/kernel_module.h"
#include "../model/bundle_loader.h"
#include <array>
#include <memory>
#include <mutex>

namespace phoenix_native::memory {
using Json = phoenix_bundle::Json;
enum class Kind : size_t {
    color,
    motion,
    workspace,
    weights,
    upload,
    history,
    views,
    samplers,
    fence,
    bridge,
    exposure,
    capture,
    guides,
    pipeline,
    transport,
    count
};
inline constexpr const char* names[]{"color",    "motion",  "workspace", "weights",  "upload",
                                     "history",  "views",   "samplers",  "fence",    "bridge",
                                     "exposure", "capture", "guides",    "pipeline", "transport"};
// Heap allocation sizes and DXGI residency answer different questions. In
// particular, heap type alone does not establish ordinary upload/readback
// residency. GPU-upload heaps explicitly support CPU writes to GPU memory.
enum class Heap : size_t { unknown, device_default, upload, readback, gpu_upload, custom, count };
inline constexpr const char* heap_names[]{"unknown",  "device_default", "upload",
                                          "readback", "gpu_upload",     "custom"};
struct Counts {
    uint64_t created = 0, released = 0, allocated = 0, freed = 0, peak = 0;
    void add(uint64_t bytes) {
        ++created;
        allocated += bytes;
        peak = (std::max)(peak, allocated - freed);
    }
    void remove(uint64_t bytes) {
        ++released;
        freed += bytes;
    }
    Json json() const {
        return {{"objects_created", created},
                {"object_release_callbacks", released},
                {"live_objects", created - released},
                {"allocation_info_bytes_created", allocated},
                {"allocation_info_bytes_released", freed},
                {"live_allocation_info_bytes", allocated - freed},
                {"peak_allocation_info_bytes", peak}};
    }
};
struct Shared {
    std::mutex mutex;
    Counts total;
    std::array<Counts, size_t(Heap::count)> heaps{};
    uint64_t next = 0, archived = 0;
};
struct Generation {
    std::shared_ptr<Shared> shared;
    uint64_t id;
    unsigned width, height;
    Counts total;
    std::array<Counts, size_t(Kind::count)> kinds{};
    std::array<Counts, size_t(Heap::count)> heaps{};
    bool constructed = false, session_destroyed = false, accounting_valid = true;
    uint64_t host_container_capacity_bytes = 0;
    std::shared_ptr<phoenix_kernel::Counters> modules;
    Generation(std::shared_ptr<Shared> shared_accounting, uint64_t number, unsigned width,
               unsigned height)
        : shared(std::move(shared_accounting)), id(number), width(width), height(height) {}
    void add(Kind kind, uint64_t bytes, Heap heap = Heap::unknown) {
        std::lock_guard<std::mutex> lock(shared->mutex);
        total.add(bytes);
        kinds.at(size_t(kind)).add(bytes);
        shared->total.add(bytes);
        heaps.at(size_t(heap)).add(bytes);
        shared->heaps.at(size_t(heap)).add(bytes);
    }
    void remove(Kind kind, uint64_t bytes, Heap heap = Heap::unknown) noexcept {
        std::lock_guard<std::mutex> lock(shared->mutex);
        total.remove(bytes);
        kinds[size_t(kind)].remove(bytes);
        shared->total.remove(bytes);
        heaps[size_t(heap)].remove(bytes);
        shared->heaps[size_t(heap)].remove(bytes);
    }
    void invalid() {
        std::lock_guard<std::mutex> lock(shared->mutex);
        accounting_valid = false;
    }
    bool reclaimable() const {
        return session_destroyed && total.created == total.released &&
               (!modules || (modules->modules_created == modules->modules_destroyed &&
                             modules->functions_created == modules->functions_destroyed &&
                             !modules->destroy_failures));
    }
    Json json() const {
        Json result = total.json(), categories = Json::object();
        for (size_t iteration_index = 0; iteration_index < size_t(Kind::count); ++iteration_index)
            categories[names[iteration_index]] = kinds[iteration_index].json();
        Json heap_allocations = Json::object();
        for (size_t index = 0; index < size_t(Heap::count); ++index)
            heap_allocations[heap_names[index]] = heaps[index].json();
        result.update(
            {{"generation_id", id},
             {"width", width},
             {"height", height},
             {"constructed", constructed},
             {"session_destroyed", session_destroyed},
             {"accounting_valid", accounting_valid},
             {"categories", categories},
             {"heap_allocations", heap_allocations},
             {"host_container_capacity_bytes_at_construction", host_container_capacity_bytes},
             {"host_container_capacity_bytes_live",
              session_destroyed ? 0 : host_container_capacity_bytes}});
        if (modules)
            result["modules"] = {{"created", modules->modules_created.load()},
                                 {"destroyed", modules->modules_destroyed.load()},
                                 {"functions_created", modules->functions_created.load()},
                                 {"functions_destroyed", modules->functions_destroyed.load()},
                                 {"destroy_failures", modules->destroy_failures.load()}};
        return result;
    }
};
class Ledger {
    std::shared_ptr<Shared> shared_ = std::make_shared<Shared>();
    std::vector<std::shared_ptr<Generation>> generations_;

  public:
    std::shared_ptr<Generation> begin(unsigned width, unsigned height) {
        std::lock_guard<std::mutex> lock(shared_->mutex);
        // Keep recent closed records and every still-live record. Cumulative
        // totals outlive compacted records and this ledger owns no GPU object.
        while (generations_.size() >= 128) {
            auto iteration_index =
                std::find_if(generations_.begin(), generations_.end(),
                             [](const auto& generation) { return generation->reclaimable(); });
            if (iteration_index == generations_.end())
                break;
            generations_.erase(iteration_index);
            ++shared_->archived;
        }
        auto value = std::make_shared<Generation>(shared_, ++shared_->next, width, height);
        generations_.push_back(value);
        return value;
    }
    Json snapshot() const {
        std::lock_guard<std::mutex> lock(shared_->mutex);
        Json rows = Json::array();
        for (const auto& generation : generations_)
            rows.push_back(generation->json());
        Json heap_allocations = Json::object();
        for (size_t index = 0; index < size_t(Heap::count); ++index)
            heap_allocations[heap_names[index]] = shared_->heaps[index].json();
        return {
            {"schema", "phoenix-owned-allocation-accounting/1"},
            {"generations_started", shared_->next},
            {"closed_records_compacted", shared_->archived},
            {"totals", shared_->total.json()},
            {"heap_allocations", heap_allocations},
            {"generations", rows},
            {"scope",
             "Owned Session/Direct Runner/Network resources, bridge and guide images, capture readbacks, descriptor heaps, pipelines and fences; final private-interface release callbacks, not GPU residency or driver heap release"},
            {"unmeasured",
             "Driver/module storage, descriptor/fence backing bytes, engine resources held by leases, diagnostic/timing resources, shared host caches and unlisted CPU containers"}};
    }
};
// First Session member: destroyed last, after network/resources/engine leases.
struct SessionLifetime {
    std::shared_ptr<Generation> generation;
    ~SessionLifetime() {
        if (generation) {
            std::lock_guard<std::mutex> lock(generation->shared->mutex);
            generation->session_destroyed = true;
        }
    }
};
inline constexpr GUID release_cookie = {
    0x7c4e8e7d, 0xda55, 0x4782, {0xb4, 0xd1, 0xc6, 0x42, 0x56, 0x03, 0x14, 0x92}};
class ReleaseToken final : public IUnknown {
    std::atomic<ULONG> refs_{1};
    std::shared_ptr<Generation> generation_;
    Kind kind_;
    uint64_t bytes_;
    Heap heap_;

  public:
    ReleaseToken(std::shared_ptr<Generation> generation, Kind kind, uint64_t bytes,
                 Heap heap = Heap::unknown)
        : generation_(std::move(generation)), kind_(kind), bytes_(bytes), heap_(heap) {
        generation_->add(kind, bytes, heap);
    }
    ~ReleaseToken() {
        generation_->remove(kind_, bytes_, heap_);
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override {
        if (!out)
            return E_POINTER;
        *out = nullptr;
        if (iid != __uuidof(IUnknown))
            return E_NOINTERFACE;
        *out = static_cast<IUnknown*>(this);
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
};
inline void track(ID3D12Object* object, const std::shared_ptr<Generation>& generation, Kind kind,
                  uint64_t bytes = 0, Heap heap = Heap::unknown) {
    if (!generation)
        return;
    Microsoft::WRL::ComPtr<IUnknown> token;
    token.Attach(new ReleaseToken(generation, kind, bytes, heap));
    if (FAILED(object->SetPrivateDataInterface(release_cookie, token.Get()))) {
        generation->invalid();
        throw std::runtime_error("phoenix allocation accounting attachment failed");
    }
}
inline void resource(ID3D12Device* device, ID3D12Resource* object,
                     const std::shared_ptr<Generation>& generation, Kind kind) {
    if (!generation)
        return;
    const auto desc = object->GetDesc();
    const auto info = device->GetResourceAllocationInfo(0, 1, &desc);
    if (!info.SizeInBytes || info.SizeInBytes == UINT64_MAX) {
        generation->invalid();
        throw std::runtime_error("phoenix invalid resource allocation size");
    }
    D3D12_HEAP_PROPERTIES properties{};
    if (FAILED(object->GetHeapProperties(&properties, nullptr))) {
        generation->invalid();
        throw std::runtime_error("phoenix resource heap classification failed");
    }
    const auto heap = properties.Type == D3D12_HEAP_TYPE_DEFAULT      ? Heap::device_default
                      : properties.Type == D3D12_HEAP_TYPE_UPLOAD     ? Heap::upload
                      : properties.Type == D3D12_HEAP_TYPE_READBACK   ? Heap::readback
                      : properties.Type == D3D12_HEAP_TYPE_GPU_UPLOAD ? Heap::gpu_upload
                                                                      : Heap::custom;
    track(object, generation, kind, info.SizeInBytes, heap);
}
}
