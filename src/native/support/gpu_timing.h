#pragma once
#include "../session/session.h"
#include <chrono>
#include <map>
#include <tuple>
#include "budget.h"
#include "timing_archive.h"

namespace phoenix_addon::phoenix {
// Separate fence: native-only lists do not submit a Session transaction.
// Slots are reserved at recording, retained through submission, and read only
// after this queue's fence completes. No CPU/GPU wait is introduced.
class GpuTiming {
    static constexpr unsigned slots = 32, stamps = 69;
    using Key = TimingArchive::Key;
    struct Range {
        int begin = -1, end = -1;
    };
    struct Slot {
        ID3D12GraphicsCommandList* list = nullptr;
        Key key{};
        unsigned count = 0, groups = 0;
        Range own, native;
        std::array<unsigned, kGroupCount * 4> group{};
        uint64_t fence = 0, epoch = 0;
    };
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12Fence> fence_;
    ComPtr<ID3D12QueryHeap> heap_;
    ComPtr<ID3D12Resource> readback_;
    std::array<Slot, slots> pending_{};
    std::shared_ptr<TimingArchive> archive_;
    unsigned current_ = slots, next_ = 0;
    uint64_t frequency_ = 0, signaled_ = 0;
    std::string settings_;
    std::string upscaler_status_;
    DepthBudget* budget_ = nullptr;
    static uint64_t now_ms() {
        return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count());
    }

    unsigned stamp() {
        auto& pending_frame = pending_[current_];
        check(pending_frame.count < stamps, "DLSS5 Phoenix timestamp capacity");
        const unsigned index = pending_frame.count++;
        pending_frame.list->EndQuery(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                                     current_ * stamps + index);
        return index;
    }

  public:
    GpuTiming(ID3D12Device* device, ID3D12CommandQueue* queue,
              std::shared_ptr<TimingArchive> archive = std::make_shared<TimingArchive>())
        : queue_(queue), archive_(std::move(archive)) {
        phoenix_native::hr(queue->GetTimestampFrequency(&frequency_));
        check(frequency_, "DLSS5 Phoenix GPU timestamp frequency missing");
        phoenix_native::hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)));
        D3D12_QUERY_HEAP_DESC query_heap_description{};
        query_heap_description.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        query_heap_description.Count = slots * stamps;
        phoenix_native::hr(device->CreateQueryHeap(&query_heap_description, IID_PPV_ARGS(&heap_)));
        readback_ =
            phoenix_native::buffer(device, uint64_t(slots) * stamps * 8, D3D12_HEAP_TYPE_READBACK,
                                   D3D12_RESOURCE_STATE_COPY_DEST);
    }
    bool matches(ID3D12CommandQueue* queue) const noexcept {
        return queue_.Get() == queue;
    }
    bool idle() const noexcept {
        for (const auto& pending_frame : pending_)
            if (pending_frame.list)
                return false;
        return true;
    }
    void restart() {
        archive_->clear();
        if (budget_)
            budget_->reset_samples();
    }
    void budget(DepthBudget* budget) noexcept {
        budget_ = budget;
    }
    const std::array<float, 120>& history() const noexcept {
        return archive_->history();
    }
    unsigned history_count() const noexcept {
        return archive_->history_count();
    }
    unsigned history_offset() const noexcept {
        return archive_->history_offset();
    }
    void settings(const std::string& key) {
        if (settings_ != key) {
            settings_ = key;
            upscaler_status_.clear();
            if (budget_)
                budget_->reset_samples();
        }
    }
    // Availability is known only after recording the upscaler. Label this
    // very frame, rather than assigning a fallback to the requested method or
    // delaying the label until the next frame. Stable status adds no JSON work.
    void upscaler_status(const std::string& status) {
        if (upscaler_status_ == status)
            return;
        upscaler_status_ = status;
        auto context = Json::parse(settings_, nullptr, false);
        if (!context.is_object())
            context = Json::object();
        context["UpscalerStatus"] = status;
        settings_ = context.dump();
        if (current_ < slots)
            std::get<4>(pending_[current_].key) = settings_;
        if (budget_)
            budget_->reset_samples();
    }
    bool begin(ID3D12GraphicsCommandList* list, unsigned mode, unsigned blocks, unsigned width,
               unsigned height) {
        check(current_ == slots, "DLSS5 Phoenix nested GPU profile");
        poll();
        if (pending_[next_].list) {
            ++archive_->dropped;
            return false;
        }
        const Key key{mode, blocks, width, height, settings_};
        current_ = next_;
        next_ = (next_ + 1) % slots;
        pending_[current_] = {};
        auto& pending_frame = pending_[current_];
        pending_frame.list = list;
        pending_frame.key = key;
        pending_frame.epoch = archive_->epoch();
        stamp();
        return true;
    }
    void own_begin() {
        if (current_ < slots)
            pending_[current_].own.begin = int(stamp());
    }
    void own_end() {
        if (current_ < slots)
            pending_[current_].own.end = int(stamp());
    }
    void native_begin() {
        if (current_ < slots)
            pending_[current_].native.begin = int(stamp());
    }
    void native_end() {
        if (current_ < slots)
            pending_[current_].native.end = int(stamp());
    }
    void group(ID3D12GraphicsCommandList* list, unsigned index) {
        if (current_ == slots)
            return;
        auto& pending_frame = pending_[current_];
        check(pending_frame.list == list && pending_frame.own.begin >= 0 &&
                  pending_frame.own.end < 0 && index == pending_frame.groups % kGroupCount &&
                  pending_frame.groups < kGroupCount * 4,
              "DLSS5 Phoenix GPU group timestamp order");
        pending_frame.group[pending_frame.groups++] = stamp();
    }
    void end() {
        if (current_ == slots)
            return;
        stamp();
        auto& pending_frame = pending_[current_];
        pending_frame.list->ResolveQueryData(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                                             current_ * stamps, pending_frame.count,
                                             readback_.Get(), uint64_t(current_) * stamps * 8);
        current_ = slots;
    }
    void abandoned(ID3D12GraphicsCommandList* list) noexcept {
        for (unsigned iteration_index = 0; iteration_index < slots; ++iteration_index)
            if (pending_[iteration_index].list == list && !pending_[iteration_index].fence) {
                pending_[iteration_index] = {};
                if (current_ == iteration_index)
                    current_ = slots;
            }
    }
    // Check before submitting; invalid reuse cannot overwrite an in-flight slot.
    void validate_submission(ID3D12CommandQueue* queue, UINT count,
                             ID3D12CommandList* const* lists) const {
        for (const auto& pending_frame : pending_)
            if (pending_frame.list && !pending_frame.fence) {
                unsigned found = 0;
                for (UINT iteration_index = 0; iteration_index < count; ++iteration_index)
                    if (lists[iteration_index] == pending_frame.list)
                        ++found;
                check(!found || (found == 1 && matches(queue) && current_ == slots),
                      "DLSS5 Phoenix timing submission identity");
            }
    }
    void submitted(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists) {
        bool found = false;
        for (const auto& pending_frame : pending_)
            if (pending_frame.list && !pending_frame.fence)
                for (UINT iteration_index = 0; iteration_index < count; ++iteration_index)
                    if (lists[iteration_index] == pending_frame.list)
                        found = true;
        if (!found)
            return;
        check(matches(queue), "DLSS5 Phoenix timing wrong queue");
        phoenix_native::hr(queue->Signal(fence_.Get(), ++signaled_));
        for (auto& pending_frame : pending_)
            if (pending_frame.list && !pending_frame.fence)
                for (UINT iteration_index = 0; iteration_index < count; ++iteration_index)
                    if (lists[iteration_index] == pending_frame.list)
                        pending_frame.fence = signaled_;
    }
    void poll() {
        const uint64_t completed = fence_->GetCompletedValue();
        check(completed != UINT64_MAX, "DLSS5 Phoenix timing device removed");
        for (unsigned iteration_index = 0; iteration_index < slots; ++iteration_index) {
            auto& pending_frame = pending_[iteration_index];
            if (!pending_frame.list || !pending_frame.fence || pending_frame.fence > completed)
                continue;
            if (pending_frame.epoch == archive_->epoch()) {
                std::array<uint64_t, stamps> values{};
                const D3D12_RANGE range{size_t(iteration_index) * stamps * 8,
                                        (size_t(iteration_index) * stamps + pending_frame.count) *
                                            8};
                void* data = nullptr;
                phoenix_native::hr(readback_->Map(0, &range, &data));
                std::memcpy(values.data(), static_cast<const uint8_t*>(data) + range.Begin,
                            pending_frame.count * 8);
                const D3D12_RANGE none{0, 0};
                readback_->Unmap(0, &none);
                auto interval_ms = [&](int begin_query, int end_query) {
                    return begin_query >= 0 && end_query > begin_query &&
                                   values[size_t(end_query)] > values[size_t(begin_query)]
                               ? double(values[size_t(end_query)] - values[size_t(begin_query)]) *
                                     1000.0 / double(frequency_)
                               : 0.0;
                };
                const double m_route = interval_ms(0, int(pending_frame.count) - 1);
                const double m_own = interval_ms(pending_frame.own.begin, pending_frame.own.end),
                             m_native =
                                 interval_ms(pending_frame.native.begin, pending_frame.native.end);
                if (std::get<0>(pending_frame.key) == 3) {
                    if (budget_)
                        budget_->sample(std::get<1>(pending_frame.key), m_own);
                }
                int before = pending_frame.own.begin;
                std::array<double, kGroupCount> group_ms{};
                for (unsigned group_index = 0; group_index < pending_frame.groups; ++group_index) {
                    group_ms[group_index % kGroupCount] +=
                        interval_ms(before, int(pending_frame.group[group_index]));
                    before = int(pending_frame.group[group_index]);
                }
                archive_->sample(pending_frame.key, pending_frame.epoch, now_ms(), m_route, m_own,
                                 m_native, pending_frame.groups / kGroupCount, group_ms);
            }
            pending_frame = {};
        }
    }
    Json report() const {
        auto context = Json::parse(settings_, nullptr, false);
        if (!context.is_object())
            context = Json::object();
        return {
            {"scope",
             "GPU command-list intervals, not whole-game frame time. Direct DLSS5 Phoenix includes all neural passes and between-pass feed/counter preparation; group times sum across passes. Legacy DLSS5 Phoenix includes Session copies; NVIDIA measures Evaluate."},
            {"frequency", frequency_},
            {"active_settings", context},
            {"epoch", archive_->epoch()},
            {"dropped", archive_->dropped},
            {"rows", archive_->rows(now_ms())}};
    }
};
}
