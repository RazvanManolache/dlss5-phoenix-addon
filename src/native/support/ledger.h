#pragma once
#include "policy.h"
#include "host_memory.h"
#include <cstdint>
#include <map>
#include <set>

namespace phoenix_addon::ledger {
// Cross-list resource state ledger.
//
// An engine can establish a resource state on one command list and consume it
// on another, so a single recording epoch is not always enough to know the
// state at an injection point. This ledger answers only what queue order
// actually proves:
//
//   * a transition recorded earlier in the SAME open recording, or
//   * the last transition on a list already submitted to the SAME queue, or
//   * the creation state, when the resource has never been transitioned.
//
// It refuses to answer while another unsubmitted recording holds a transition
// of that resource, because that recording may be submitted first. Nothing is
// inferred from a resource's name, usage or a driver convention.
enum class Basis { unknown, creation, current_recording, submitted_transition };

class Ledger {
    struct Resource {
        uint32_t created_state = 0;
        uint32_t committed_state = 0;
        uint64_t committed_queue = 0;
        uint64_t transitions = 0;
        unsigned open_recordings = 0; // unsubmitted recordings holding a transition
        bool created = false;
        bool committed = false;
    };
    host_memory::Map<uint64_t, Resource, host_memory::Tag::resources> resources_;
    using Transitions = host_memory::Map<uint64_t, uint32_t, host_memory::Tag::transitions>;
    host_memory::Map<uint64_t, Transitions, host_memory::Tag::recordings> open_;
    uint64_t forgotten_ = 0;
    uint64_t destroyed_pending_entries_ = 0;

    void forget_resource(uint64_t resource) {
        for (auto list = open_.begin(); list != open_.end();) {
            destroyed_pending_entries_ += list->second.erase(resource);
            list = list->second.empty() ? open_.erase(list) : std::next(list);
        }
        resources_.erase(resource);
    }

    void release(uint64_t list) {
        const auto recording_entry = open_.find(list);
        if (recording_entry == open_.end())
            return;
        for (const auto& entry : recording_entry->second) {
            const auto resource = resources_.find(entry.first);
            if (resource != resources_.end() && resource->second.open_recordings)
                --resource->second.open_recordings;
        }
        open_.erase(recording_entry);
    }

  public:
    struct Answer {
        bool known = false;
        uint32_t state = 0;
        Basis basis = Basis::unknown;
    };
    void created(uint64_t resource, uint32_t state) {
        check(resource != 0, "phoenix ledger null resource");
        if (resources_.count(resource))
            forget_resource(resource);
        auto& entry = resources_[resource];
        // A recycled handle must never inherit the previous resource's history.
        entry = {};
        entry.created = true;
        entry.created_state = state;
    }
    void destroyed(uint64_t resource) {
        forget_resource(resource);
    }
    // A recording epoch begins: anything this list had recorded and not
    // submitted can never execute.
    void reset(uint64_t list) {
        release(list);
    }
    void destroyed_list(uint64_t list) {
        release(list);
    }
    void destroyed_queue(uint64_t queue) {
        for (auto& entry : resources_)
            if (entry.second.committed_queue == queue) {
                entry.second.committed = false;
                entry.second.committed_queue = 0;
            }
    }
    void transition(uint64_t list, uint64_t resource, uint32_t state) {
        check(list != 0 && resource != 0, "phoenix ledger null transition operand");
        if (open_.size() > 4096 || resources_.size() > 262144) {
            ++forgotten_;
            return;
        }
        auto& recorded = open_[list];
        auto& entry = resources_[resource];
        if (recorded.find(resource) == recorded.end())
            ++entry.open_recordings;
        recorded[resource] = state;
        ++entry.transitions;
    }
    // Called after the real ExecuteCommandLists returned for this list.
    void submitted(uint64_t list, uint64_t queue) {
        check(list != 0 && queue != 0, "phoenix ledger null submission operand");
        const auto recording_entry = open_.find(list);
        if (recording_entry == open_.end())
            return;
        for (const auto& entry : recording_entry->second) {
            auto& resource = resources_[entry.first];
            resource.committed = true;
            resource.committed_state = entry.second;
            resource.committed_queue = queue;
            if (resource.open_recordings)
                --resource.open_recordings;
        }
        open_.erase(recording_entry);
    }
    // `recording_list` is the list currently being recorded into and `queue` is
    // the queue it is proven to be submitted on.
    Answer state(uint64_t resource, uint64_t recording_list, uint64_t queue) const {
        Answer answer;
        const auto entry = resources_.find(resource);
        if (entry == resources_.end())
            return answer;
        const auto recording = open_.find(recording_list);
        if (recording != open_.end()) {
            const auto own = recording->second.find(resource);
            if (own != recording->second.end())
                return {true, own->second, Basis::current_recording};
        }
        unsigned others = entry->second.open_recordings;
        if (recording != open_.end() && recording->second.count(resource))
            --others;
        if (others)
            return answer; // another unsubmitted recording may run first
        if (entry->second.committed) {
            if (entry->second.committed_queue != queue)
                return answer;
            return {true, entry->second.committed_state, Basis::submitted_transition};
        }
        if (entry->second.created && entry->second.transitions == 0)
            return {true, entry->second.created_state, Basis::creation};
        return answer;
    }
    // Read-only refusal evidence: names WHY state() answers unknown for a
    // resource, so a live refusal carries the disambiguation instead of
    // needing a follow-up run. No mutation, no inference.
    struct Explanation {
        bool tracked = false;
        bool open_in_this_list = false;
        unsigned other_open_recordings = 0;
        bool committed = false;
        bool committed_queue_matches = false;
        bool created_never_transitioned = false;
        uint64_t transitions = 0;
        uint64_t open_recording_lists = 0;
    };
    Explanation explain(uint64_t resource, uint64_t recording_list, uint64_t queue) const {
        Explanation out;
        out.open_recording_lists = open_.size();
        const auto entry = resources_.find(resource);
        if (entry == resources_.end())
            return out;
        out.tracked = true;
        out.transitions = entry->second.transitions;
        const auto recording = open_.find(recording_list);
        if (recording != open_.end() && recording->second.count(resource))
            out.open_in_this_list = true;
        out.other_open_recordings = entry->second.open_recordings;
        if (out.open_in_this_list && out.other_open_recordings)
            --out.other_open_recordings;
        out.committed = entry->second.committed;
        out.committed_queue_matches =
            entry->second.committed && entry->second.committed_queue == queue;
        out.created_never_transitioned = entry->second.created && entry->second.transitions == 0;
        return out;
    }
    uint64_t tracked_resources() const noexcept {
        return resources_.size();
    }
    uint64_t open_recordings() const noexcept {
        return open_.size();
    }
    uint64_t forgotten() const noexcept {
        return forgotten_;
    }
    uint64_t destroyed_pending_entries() const noexcept {
        return destroyed_pending_entries_;
    }
    uint64_t transition_entries() const noexcept {
        uint64_t count = 0;
        for (const auto& list : open_)
            count += list.second.size();
        return count;
    }
};
inline const char* basis_name(Basis basis) noexcept {
    switch (basis) {
    case Basis::creation:
        return "creation_state_never_transitioned";
    case Basis::current_recording:
        return "transition_in_current_recording";
    case Basis::submitted_transition:
        return "last_transition_submitted_earlier_on_same_queue";
    default:
        return "unknown";
    }
}
} // namespace production_game41::ledger
