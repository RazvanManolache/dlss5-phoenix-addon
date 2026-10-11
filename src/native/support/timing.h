#pragma once
#include "../session/session.h"

namespace phoenix_addon::timing {
// GPU timestamps recorded into the engine's own command list, around exactly
// the work being compared. In paired mode the reconstructed network and the
// untouched original are measured in the same frame, on the same queue, one
// after the other, so the two numbers share a timing scope by construction.
// Nothing is ever waited on: a frame's timestamps are read only after the
// submission fence that carried them has completed.
//
// Layout per frame, contiguous so that no unwritten query is ever resolved:
//   0                candidate begin
//   1 .. groups      one per completed natural dispatch group, in order
//   groups+1         candidate end
//   groups+2, +3     original begin and end, paired mode only
class Timestamps {
    static constexpr unsigned frames = 32;     // ring of frame slots
    static constexpr unsigned max_groups = 20; // natural dispatch groups
    static constexpr unsigned per_frame = max_groups + 4;

    ComPtr<ID3D12QueryHeap> heap_;
    ComPtr<ID3D12Resource> readback_;
    uint64_t frequency_ = 0;
    unsigned next_ = 0;
    struct Pending {
        unsigned base = 0, count = 0, groups = 0;
        uint64_t fence = 0;
        bool armed = false, paired = false;
    };
    std::array<Pending, frames> pending_{};
    unsigned writing_ = UINT_MAX; // slot index currently being recorded
    unsigned written_ = 0;        // stamps written into that slot so far

    uint64_t candidate_samples_ = 0, original_samples_ = 0, dropped_ = 0;
    double candidate_total_ = 0, original_total_ = 0;
    double candidate_max_ = 0, original_max_ = 0, candidate_min_ = 0, original_min_ = 0;
    std::array<double, max_groups> group_total_{};
    std::array<uint64_t, max_groups> group_samples_{};
    unsigned groups_seen_ = 0;

    double milliseconds(uint64_t begin, uint64_t end) const {
        if (!frequency_ || end <= begin)
            return 0.0;
        return double(end - begin) * 1000.0 / double(frequency_);
    }
    void write(ID3D12GraphicsCommandList* cmd, unsigned index) noexcept {
        cmd->EndQuery(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, pending_[writing_].base + index);
    }

  public:
    Timestamps(ID3D12Device* device, ID3D12CommandQueue* queue) {
        check(device && queue, "phoenix timing requires a device and queue");
        phoenix_native::hr(queue->GetTimestampFrequency(&frequency_));
        check(frequency_ != 0, "phoenix timing frequency unavailable");
        D3D12_QUERY_HEAP_DESC desc{};
        desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        desc.Count = frames * per_frame;
        phoenix_native::hr(device->CreateQueryHeap(&desc, IID_PPV_ARGS(&heap_)));
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = UINT64(frames) * per_frame * sizeof(uint64_t);
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        phoenix_native::hr(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                           D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                           IID_PPV_ARGS(&readback_)));
    }
    Timestamps(const Timestamps&) = delete;
    uint64_t frequency() const noexcept {
        return frequency_;
    }
    bool recording() const noexcept {
        return writing_ != UINT_MAX;
    }

    // Opens a frame and writes the candidate begin stamp. Returns false when
    // the ring still holds an unread frame; that frame is counted, not guessed.
    bool begin_frame(ID3D12GraphicsCommandList* cmd) noexcept {
        if (pending_[next_].armed) {
            ++dropped_;
            writing_ = UINT_MAX;
            return false;
        }
        writing_ = next_;
        next_ = (next_ + 1) % frames;
        pending_[writing_] = {writing_ * per_frame, 0, 0, 0, false, false};
        written_ = 0;
        write(cmd, 0);
        written_ = 1;
        return true;
    }
    void group(ID3D12GraphicsCommandList* cmd) noexcept {
        if (writing_ == UINT_MAX)
            return;
        auto& slot = pending_[writing_];
        if (slot.groups >= max_groups)
            return;
        write(cmd, 1 + slot.groups);
        ++slot.groups;
        written_ = 1 + slot.groups;
    }
    void candidate_end(ID3D12GraphicsCommandList* cmd) noexcept {
        if (writing_ == UINT_MAX)
            return;
        write(cmd, 1 + pending_[writing_].groups);
        written_ = 2 + pending_[writing_].groups;
    }
    void original_begin(ID3D12GraphicsCommandList* cmd) noexcept {
        if (writing_ == UINT_MAX)
            return;
        write(cmd, 2 + pending_[writing_].groups);
        written_ = 3 + pending_[writing_].groups;
    }
    void original_end(ID3D12GraphicsCommandList* cmd) noexcept {
        if (writing_ == UINT_MAX)
            return;
        write(cmd, 3 + pending_[writing_].groups);
        written_ = 4 + pending_[writing_].groups;
        pending_[writing_].paired = true;
    }
    // Resolves exactly the stamps that were written. Returns the slot, or
    // UINT_MAX when no frame was open.
    unsigned resolve(ID3D12GraphicsCommandList* cmd) noexcept {
        if (writing_ == UINT_MAX)
            return UINT_MAX;
        auto& slot = pending_[writing_];
        slot.count = written_;
        cmd->ResolveQueryData(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, slot.base, slot.count,
                              readback_.Get(), UINT64(slot.base) * sizeof(uint64_t));
        const auto opened = writing_;
        writing_ = UINT_MAX;
        return opened;
    }
    // Called after the real submission, with the fence value that retires it.
    void submitted(unsigned slot, uint64_t fence) noexcept {
        if (slot >= frames || !pending_[slot].count)
            return;
        pending_[slot].fence = fence;
        pending_[slot].armed = true;
    }
    void poll(uint64_t completed_fence) {
        for (auto& slot : pending_) {
            if (!slot.armed || slot.fence == 0 || completed_fence < slot.fence)
                continue;
            std::array<uint64_t, per_frame> values{};
            const D3D12_RANGE read{size_t(slot.base) * sizeof(uint64_t),
                                   size_t(slot.base + slot.count) * sizeof(uint64_t)};
            void* data = nullptr;
            if (SUCCEEDED(readback_->Map(0, &read, &data)) && data) {
                std::memcpy(values.data(), static_cast<const uint8_t*>(data) + read.Begin,
                            size_t(slot.count) * sizeof(uint64_t));
                const D3D12_RANGE written{0, 0};
                readback_->Unmap(0, &written);
                const unsigned end = 1 + slot.groups;
                if (slot.count > end) {
                    const auto duration_ms = milliseconds(values[0], values[end]);
                    if (duration_ms > 0) {
                        if (!candidate_samples_ || duration_ms < candidate_min_)
                            candidate_min_ = duration_ms;
                        if (duration_ms > candidate_max_)
                            candidate_max_ = duration_ms;
                        candidate_total_ += duration_ms;
                        ++candidate_samples_;
                    }
                    uint64_t previous = values[0];
                    for (unsigned group_index = 0; group_index < slot.groups; ++group_index) {
                        const auto step = milliseconds(previous, values[1 + group_index]);
                        if (values[1 + group_index] > previous) {
                            group_total_[group_index] += step;
                            ++group_samples_[group_index];
                        }
                        previous = values[1 + group_index];
                    }
                    if (slot.groups > groups_seen_)
                        groups_seen_ = slot.groups;
                }
                if (slot.paired && slot.count >= end + 3) {
                    const auto duration_ms = milliseconds(values[end + 1], values[end + 2]);
                    if (duration_ms > 0) {
                        if (!original_samples_ || duration_ms < original_min_)
                            original_min_ = duration_ms;
                        if (duration_ms > original_max_)
                            original_max_ = duration_ms;
                        original_total_ += duration_ms;
                        ++original_samples_;
                    }
                }
            }
            slot.armed = false;
        }
    }
    void restart() noexcept {
        candidate_samples_ = original_samples_ = 0;
        candidate_total_ = original_total_ = 0;
        candidate_max_ = original_max_ = candidate_min_ = original_min_ = 0;
        group_total_ = {};
        group_samples_ = {};
    }
    Json report() const {
        auto mean = [](double total, uint64_t count) {
            return count ? total / double(count) : 0.0;
        };
        Json per_group = Json::array();
        for (unsigned group_index = 0; group_index < groups_seen_; ++group_index)
            per_group.push_back(
                Json{{"group", group_index},
                     {"samples", group_samples_[group_index]},
                     {"mean_ms", mean(group_total_[group_index], group_samples_[group_index])}});
        return {
            {"timestamp_frequency", frequency_},
            {"scope",
             "GPU timestamps around the recorded work inside the engine's own command list on its own queue"},
            {"candidate_samples", candidate_samples_},
            {"candidate_mean_ms", mean(candidate_total_, candidate_samples_)},
            {"candidate_min_ms", candidate_min_},
            {"candidate_max_ms", candidate_max_},
            {"original_samples", original_samples_},
            {"original_mean_ms", mean(original_total_, original_samples_)},
            {"original_min_ms", original_min_},
            {"original_max_ms", original_max_},
            {"frames_without_a_free_slot", dropped_},
            {"natural_dispatch_groups", groups_seen_},
            {"group_mean_ms", per_group},
        };
    }
};
} // namespace production_game41::timing
