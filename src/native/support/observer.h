#pragma once
#include "../session/session.h"
#include "../../../third_party/minhook/include/MinHook.h"
#include "hook_batch.h"
#include <string>
#include <cstdio>
#include <atomic>
#include <map>

namespace phoenix_addon::observation {
// Slots verified against Windows SDK 10.0.26100 ID3D12GraphicsCommandList7Vtbl.
// Only barriers observed in this recording epoch establish resource state.
struct Root {
    unsigned kind = 0;
    uint64_t address = 0, mask = 0;
    std::array<uint32_t, 64> constants{};
};
struct Bindings {
    ComPtr<ID3D12PipelineState> pipeline;
    std::array<ComPtr<ID3D12RootSignature>, 2> signatures;
    std::array<std::array<Root, 64>, 2> roots;
    host_memory::Vector<ComPtr<ID3D12DescriptorHeap>, host_memory::Tag::heap_refs> heaps;
    bool heaps_known = false;
    void restore(ID3D12GraphicsCommandList* cmd) const {
        std::vector<ID3D12DescriptorHeap*> raw;
        for (const auto& heap : heaps)
            raw.push_back(heap.Get());
        cmd->SetDescriptorHeaps(UINT(raw.size()), raw.data());
        if (pipeline)
            cmd->SetPipelineState(pipeline.Get());
        for (unsigned graphics = 0; graphics < 2; ++graphics) {
            if (!signatures[graphics])
                continue;
            if (graphics)
                cmd->SetGraphicsRootSignature(signatures[graphics].Get());
            else
                cmd->SetComputeRootSignature(signatures[graphics].Get());
            for (unsigned slot_index = 0; slot_index < 64; ++slot_index) {
                const auto& root = roots[graphics][slot_index];
                switch (root.kind) {
                case 1:
                    if (graphics)
                        cmd->SetGraphicsRootDescriptorTable(slot_index, {root.address});
                    else
                        cmd->SetComputeRootDescriptorTable(slot_index, {root.address});
                    break;
                case 2:
                    for (unsigned word_index = 0; word_index < 64; ++word_index)
                        if (root.mask & (uint64_t(1) << word_index)) {
                            if (graphics)
                                cmd->SetGraphicsRoot32BitConstant(
                                    slot_index, root.constants[word_index], word_index);
                            else
                                cmd->SetComputeRoot32BitConstant(
                                    slot_index, root.constants[word_index], word_index);
                        }
                    break;
                case 3:
                    if (graphics)
                        cmd->SetGraphicsRootConstantBufferView(slot_index, root.address);
                    else
                        cmd->SetComputeRootConstantBufferView(slot_index, root.address);
                    break;
                case 4:
                    if (graphics)
                        cmd->SetGraphicsRootShaderResourceView(slot_index, root.address);
                    else
                        cmd->SetComputeRootShaderResourceView(slot_index, root.address);
                    break;
                case 5:
                    if (graphics)
                        cmd->SetGraphicsRootUnorderedAccessView(slot_index, root.address);
                    else
                        cmd->SetComputeRootUnorderedAccessView(slot_index, root.address);
                    break;
                }
            }
        }
    }
};
// Bits recorded alongside `unsupported` so a rejection names the construct that
// caused it instead of being an opaque refusal.
enum Unsupported : uint32_t {
    unsupported_split_barrier = 1u << 0,
    unsupported_wildcard_aliasing_barrier = 1u << 12,
    unsupported_null_barrier_resource = 1u << 1,
    unsupported_barrier_state_conflict = 1u << 2,
    unsupported_enhanced_barrier = 1u << 3,
    unsupported_execute_bundle = 1u << 4,
    unsupported_execute_indirect = 1u << 5,
    unsupported_state_object = 1u << 6,
    unsupported_protected_session = 1u << 7,
    unsupported_meta_command = 1u << 8,
    unsupported_observer_exception = 1u << 9,
    unsupported_close_failed = 1u << 10,
    unsupported_reset_failed = 1u << 11,
};
struct Command {
    Bindings bindings;
    host_memory::Map<ID3D12Resource*, D3D12_RESOURCE_STATES, host_memory::Tag::epoch_states> states;
    host_memory::Map<ID3D12Resource*, D3D12_RESOURCE_STATES, host_memory::Tag::epoch_states>
        depth_plane_states;
    bool unsupported = false, render_pass = false, predicated = false, closed = false,
         epoch_known = false;
    uint32_t unsupported_mask = 0;
    uint32_t split_barriers = 0, aliasing_barriers = 0, wildcard_aliasing_barriers = 0;
    // Depth-copy diagnostics: planar depth transitions recorded in-list vs
    // rejected by the shape filter. A live run with seen==0 means depth is
    // never transitioned in the claimed list; filtered>0 names the filter.
    uint32_t planar_transitions_seen = 0, planar_transitions_filtered = 0;
    // Bounded evidence of which resources an aliasing barrier actually named.
    host_memory::Vector<std::pair<ID3D12Resource*, ID3D12Resource*>, host_memory::Tag::aliasing>
        aliasing;
    // Diagnosis only: states this epoch established before a wildcard aliasing
    // barrier discarded them. `resolve` never reads this.
    host_memory::Map<ID3D12Resource*, D3D12_RESOURCE_STATES, host_memory::Tag::discarded_states>
        discarded_states;
    // Resources an aliasing barrier named in this epoch. A named operand is
    // positive evidence that THAT resource was aliased over, so no older
    // ledger answer may be used for it.
    host_memory::Set<ID3D12Resource*, host_memory::Tag::aliasing> aliased;
    // Last explicitly deactivated operands, unlike the historical alias set.
    host_memory::Set<ID3D12Resource*, host_memory::Tag::aliasing> deactivated;
    // ExecuteIndirect bookkeeping. `accepted` counts calls whose signature was
    // observed at creation and proved binding neutral; the two refusal counts
    // and the type mask say why the others were refused, so a rejection names
    // the argument types involved instead of being opaque.
    uint32_t indirect_accepted = 0, indirect_unknown_signature = 0, indirect_rejected_signature = 0;
    uint32_t indirect_rejected_type_mask = 0;
};
inline void mark(Command& command, uint32_t bit) noexcept {
    command.unsupported = true;
    command.unsupported_mask |= bit;
}
// An ExecuteIndirect is a draw or dispatch, not a transition: D3D12 does not
// change any resource's tracked state because of one. What it can change is
// BINDINGS -- several indirect argument types write root arguments or IA views
// straight from GPU memory we never observe, and those writes persist on the
// list afterwards, so the recording's bindings would no longer be the ones we
// captured and could restore. A signature built only from whole draw/dispatch
// arguments writes no binding at all, which is the one case that is safe.
// Everything else stays refused, and the refusal records which argument types
// it saw so the next capture says exactly what a wider contract would need.
inline bool binding_neutral(D3D12_INDIRECT_ARGUMENT_TYPE type) noexcept {
    return type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW ||
           type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED ||
           type == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH ||
           type == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_RAYS ||
           type == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
}
struct SignatureInfo {
    bool neutral = false;
    uint32_t arguments = 0, type_mask = 0, byte_stride = 0;
};
// A resource whose CreateCommittedResource call we saw ourselves.
//
// This exists because ReShade's own init_resource event is not guaranteed to
// reach us: a host may create a D3D12 device before ReShade loads, and
// resources made on it never generate an event. RE Requiem is exactly that
// case -- its NR output reports `output_creation_state_known: false` while
// `output_transitions_seen_anywhere` is 0.
//
// A committed resource owns its own implicit heap, so no other resource can
// alias its memory. That is what makes a creation-state answer admissible at
// all, and only for the narrow shape below. Everything outside it is recorded
// as ineligible rather than omitted, so a refusal can say which rule failed.
enum Ineligible : uint32_t {
    ineligible_heap_type = 1u << 0,
    ineligible_shared_heap = 1u << 1,
    ineligible_not_texture2d = 1u << 2,
    ineligible_simultaneous_access = 1u << 3,
    ineligible_initial_state = 1u << 4,
    ineligible_creation_failed = 1u << 5,
};
struct CommittedInfo {
    uint32_t initial_state = 0;
    uint32_t ineligible_mask = 0;
    bool eligible = false;
    // True when the resource's own Release is observed, so its record dies
    // with it. False only when registration relied on an external destruction
    // feed (the ReShade host's destroy_resource event) instead.
    bool release_observed = false;
    // Set the moment a barrier on ANY command list NAMES this resource,
    // observed recording or not. The barrier detour is a vtable patch, so it
    // sees every list of that implementation. This ends the creation answer.
    bool touched_by_named_barrier = false;
    // Set by a wildcard aliasing barrier, which names no resource. Recorded
    // but deliberately NOT disqualifying, and only for a resource that passed
    // the eligibility rules below -- see `committed_creation_state`.
    bool seen_wildcard_aliasing = false;
};
struct Observer {
    std::recursive_mutex mutex;
    host_memory::Map<ID3D12GraphicsCommandList*, Command, host_memory::Tag::commands> commands;
    // Command signatures observed at creation. A signature absent here was
    // created before the device hook was installed and is therefore unknown,
    // which is refused separately from one that was seen and rejected.
    host_memory::Map<ID3D12CommandSignature*, SignatureInfo, host_memory::Tag::command_signatures>
        signatures;
    host_memory::Map<ID3D12Resource*, CommittedInfo, host_memory::Tag::committed_resources>
        committed;
    // Set when the registry hit its cap, which means a creation we needed may
    // be missing. It disables the creation answer outright rather than letting
    // an absent record read as "never created".
    bool committed_overflow = false;
    // IUnknown::Release hooks, one per distinct ID3D12Resource implementation
    // vtable seen. See install_resource_release: without an exact destruction
    // feed, a dead resource's committed record lingers and a recycled pointer
    // created through an unhooked path would inherit a creation answer it
    // never earned. Each slot's detour calls only its own original, so the
    // fast path (a Release that does not destroy anything) is one indirect
    // call plus a comparison, with no lock and no map lookup.
    static constexpr unsigned kResourceReleaseVtables = 8;
    std::array<void*, kResourceReleaseVtables> resource_release_targets{};
    std::array<std::atomic<void*>, kResourceReleaseVtables> resource_release_originals{};
    // Slots handed out (under lock) versus slots published (only after the
    // hook is enabled). They differ while an install is in flight; published
    // slots are the only ones scanned, and an unpublished slot reads null and
    // can never match. A failed install wastes its claimed slot, which is
    // bounded by the table size and fails closed via the overflow flag.
    unsigned resource_release_claimed = 0;
    unsigned resource_release_hooked = 0;
    bool resource_release_overflow = false;
    // Set by a host that feeds destruction externally (ReShade's
    // destroy_resource event). When true, a resource may register even if its
    // Release cannot be hooked, because the external feed still retires its
    // record. The ASI host leaves this false and is therefore strict: no
    // Release hook, no registration, no answer.
    bool external_destruction_feed = false;
    std::atomic<uint64_t> resource_final_releases{0};
    std::atomic<uint64_t> command_final_releases{0};
    std::array<void*, 42> device_targets{}, device_originals{};
    std::array<void*, 81> targets{}, originals{};
    void (*reset_callback)(ID3D12GraphicsCommandList*, bool) = nullptr;
    void (*close_callback)(ID3D12GraphicsCommandList*, bool) = nullptr;
};
inline Observer& observer() {
    static auto* observer_state = new Observer;
    return *observer_state;
}
template <class F> void change(ID3D12GraphicsCommandList* cmd, F action) noexcept {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    const auto command_entry = observer_state.commands.find(cmd);
    if (command_entry == observer_state.commands.end())
        return;
    try {
        action(command_entry->second);
    } catch (...) {
        mark(command_entry->second, unsupported_observer_exception);
    }
}
template <class F> F original(unsigned slot) {
    return reinterpret_cast<F>(observer().originals[slot]);
}
template <class F> F device_original(unsigned slot) {
    return reinterpret_cast<F>(observer().device_originals[slot]);
}
inline HRESULT STDMETHODCALLTYPE close(ID3D12GraphicsCommandList* cmd) {
    const auto result = original<decltype(&close)>(9)(cmd);
    change(cmd, [&](Command& command) {
        command.closed = SUCCEEDED(result);
        if (FAILED(result)) {
            mark(command, unsupported_close_failed);
            return;
        }
        // Closed lists cannot be injected into. A later successful Reset
        // establishes fresh bindings; existing snapshots own their own leases.
        command.bindings.pipeline.Reset();
        for (auto& signature : command.bindings.signatures)
            signature.Reset();
        command.bindings.heaps.clear();
    });
    if (observer().close_callback)
        observer().close_callback(cmd, SUCCEEDED(result));
    return result;
}
inline HRESULT STDMETHODCALLTYPE reset(ID3D12GraphicsCommandList* cmd,
                                       ID3D12CommandAllocator* allocator,
                                       ID3D12PipelineState* pso) {
    const auto result = original<decltype(&reset)>(10)(cmd, allocator, pso);
    change(cmd, [&](Command& command) {
        if (SUCCEEDED(result)) {
            command = {};
            command.epoch_known = true;
            command.bindings.pipeline = pso;
        } else
            mark(command, unsupported_reset_failed);
    });
    if (observer().reset_callback)
        observer().reset_callback(cmd, SUCCEEDED(result));
    return result;
}
inline void STDMETHODCALLTYPE clear(ID3D12GraphicsCommandList* cmd, ID3D12PipelineState* pso) {
    change(cmd, [&](Command& command) {
        command.bindings = {};
        command.bindings.pipeline = pso;
        command.predicated = false;
    });
    original<decltype(&clear)>(11)(cmd, pso);
}
inline void STDMETHODCALLTYPE pipeline(ID3D12GraphicsCommandList* cmd, ID3D12PipelineState* pso) {
    change(cmd, [&](Command& command) { command.bindings.pipeline = pso; });
    original<decltype(&pipeline)>(25)(cmd, pso);
}
// Every barrier on every list, whether or not the list is an observed
// recording. Only resources already in the committed registry are looked up,
// so this stays O(1) per operand and bounded by that registry.
inline void mark_barrier_touched(UINT count, const D3D12_RESOURCE_BARRIER* list) noexcept {
    auto& observer_state = observer();
    if (observer_state.committed.empty() || !list)
        return;
    auto touch = [&](ID3D12Resource* resource) {
        if (!resource)
            return;
        const auto resource_state_entry = observer_state.committed.find(resource);
        if (resource_state_entry != observer_state.committed.end())
            resource_state_entry->second.touched_by_named_barrier = true;
    };
    for (UINT slot_index = 0; slot_index < count; ++slot_index) {
        const auto& binding = list[slot_index];
        if (binding.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION)
            touch(binding.Transition.pResource);
        else if (binding.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV)
            touch(binding.UAV.pResource);
        else if (binding.Type == D3D12_RESOURCE_BARRIER_TYPE_ALIASING) {
            touch(binding.Aliasing.pResourceBefore);
            touch(binding.Aliasing.pResourceAfter);
            // A wildcard operand names nothing. Aliasing redefines memory that
            // two resources SHARE, and a committed resource owns its own
            // implicit heap: no placed or reserved resource can be created
            // over it, because its heap has no handle to place into. So a
            // wildcard aliasing barrier cannot redefine a private committed
            // resource's contents. Shared and cross-adapter heaps are the
            // cases where another party could hold the memory, and those are
            // already rejected at creation. This is recorded rather than
            // ignored so a summary can show how often it happened.
            if (!binding.Aliasing.pResourceBefore || !binding.Aliasing.pResourceAfter)
                for (auto& entry : observer_state.committed)
                    entry.second.seen_wildcard_aliasing = true;
        }
    }
}
inline void STDMETHODCALLTYPE barriers(ID3D12GraphicsCommandList* cmd, UINT count,
                                       const D3D12_RESOURCE_BARRIER* values) {
    {
        auto& observer_state = observer();
        std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
        mark_barrier_touched(count, values);
    }
    change(cmd, [&](Command& command) {
        for (UINT slot_index = 0; slot_index < count; ++slot_index) {
            const auto& binding = values[slot_index];
            if (binding.Flags != D3D12_RESOURCE_BARRIER_FLAG_NONE) {
                // A split BEGIN/END pair leaves the resource unusable between
                // halves; only that resource's record is dropped, since a split
                // barrier cannot change any other resource's tracked state.
                ++command.split_barriers;
                mark(command, unsupported_split_barrier);
                if (binding.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION) {
                    if (binding.Transition.pResource) {
                        command.states.erase(binding.Transition.pResource);
                        command.depth_plane_states.erase(binding.Transition.pResource);
                    }
                } else if (binding.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV) {
                    if (binding.UAV.pResource) {
                        command.states.erase(binding.UAV.pResource);
                        command.depth_plane_states.erase(binding.UAV.pResource);
                    }
                } else if (binding.Type == D3D12_RESOURCE_BARRIER_TYPE_ALIASING) {
                    if (binding.Aliasing.pResourceBefore) {
                        command.states.erase(binding.Aliasing.pResourceBefore);
                        command.depth_plane_states.erase(binding.Aliasing.pResourceBefore);
                    }
                    if (binding.Aliasing.pResourceAfter) {
                        command.states.erase(binding.Aliasing.pResourceAfter);
                        command.depth_plane_states.erase(binding.Aliasing.pResourceAfter);
                    }
                }
                continue;
            }
            if (binding.Type == D3D12_RESOURCE_BARRIER_TYPE_ALIASING) {
                ++command.aliasing_barriers;
                auto* before = binding.Aliasing.pResourceBefore;
                auto* after = binding.Aliasing.pResourceAfter;
                if (command.aliasing.size() < 16)
                    command.aliasing.emplace_back(before, after);
                // A named operand is positive evidence that THAT resource was
                // aliased over: it loses its contents and must be reinitialized.
                // Record it so no older ledger answer can be used for it again.
                if (before) {
                    command.states.erase(before);
                    command.depth_plane_states.erase(before);
                    command.aliased.insert(before);
                }
                if (after) {
                    command.states.erase(after);
                    command.depth_plane_states.erase(after);
                    command.aliased.insert(after);
                }
                if (before)
                    command.deactivated.insert(before);
                if (after)
                    command.deactivated.erase(after);
                // A NULL operand is the "any resource" wildcard: some unnamed
                // placed resource changed occupancy. Drop every tracked epoch
                // state, so only a transition observed AFTER this point can be
                // used. Aliasing redefines contents, not D3D12 tracked state,
                // so the queue-ordered ledger answer stays admissible; see
                // aliasing_permits.
                if (!before || !after) {
                    ++command.wildcard_aliasing_barriers;
                    for (const auto& entry : command.states)
                        command.discarded_states[entry.first] = entry.second;
                    // Depth plane observations survive on purpose: aliasing
                    // redefines memory occupancy, never D3D12 tracked state,
                    // so a preserved StateBefore stays a legal copy barrier.
                    command.states.clear();
                }
                continue;
            }
            if (binding.Type != D3D12_RESOURCE_BARRIER_TYPE_TRANSITION)
                continue;
            auto* resource = binding.Transition.pResource;
            if (!resource) {
                mark(command, unsupported_null_barrier_resource);
                continue;
            }
            const auto description = resource->GetDesc();
            if (description.MipLevels == 1 && description.DepthOrArraySize == 1 &&
                (description.Format == DXGI_FORMAT_R32G8X24_TYPELESS ||
                 description.Format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT)) {
                const auto plane = binding.Transition.Subresource;
                if (plane == 0 || plane == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) {
                    const auto prior = command.depth_plane_states.find(resource);
                    if (prior != command.depth_plane_states.end() &&
                        prior->second != binding.Transition.StateBefore)
                        mark(command, unsupported_barrier_state_conflict);
                    command.depth_plane_states[resource] = binding.Transition.StateAfter;
                    ++command.planar_transitions_seen;
                } else if (plane != 1)
                    command.depth_plane_states.erase(resource);
                // Stencil transitions must not erase the independently observed depth state.
                // Full-resource callers only receive an all-planes observation.
                if (plane == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)
                    command.states[resource] = binding.Transition.StateAfter;
                else
                    command.states.erase(resource);
                continue;
            }
            if ((description.Format == DXGI_FORMAT_R32G8X24_TYPELESS ||
                 description.Format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT) &&
                (description.MipLevels != 1 || description.DepthOrArraySize != 1))
                ++command.planar_transitions_filtered;
            if (description.MipLevels != 1 || description.DepthOrArraySize != 1 ||
                (binding.Transition.Subresource != 0 &&
                 binding.Transition.Subresource != D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)) {
                command.states.erase(resource);
                continue;
            }
            const auto before = command.states.find(resource);
            if (before != command.states.end() && before->second != binding.Transition.StateBefore)
                mark(command, unsupported_barrier_state_conflict);
            command.states[resource] = binding.Transition.StateAfter;
        }
    });
    original<decltype(&barriers)>(26)(cmd, count, values);
}
inline void STDMETHODCALLTYPE heaps(ID3D12GraphicsCommandList* cmd, UINT count,
                                    ID3D12DescriptorHeap* const* values) {
    change(cmd, [&](Command& command) {
        check(count <= 2, "phoenix descriptor heap count");
        bool changed = count != command.bindings.heaps.size();
        for (UINT slot_index = 0; !changed && slot_index < count; ++slot_index)
            changed = command.bindings.heaps[slot_index].Get() != values[slot_index];
        command.bindings.heaps.clear();
        for (UINT slot_index = 0; slot_index < count; ++slot_index)
            command.bindings.heaps.emplace_back(values[slot_index]);
        command.bindings.heaps_known = true;
        // Heap changes invalidate tables. Subsequent observed table setters
        // establish precisely the bindings that need to be restored.
        if (changed)
            for (auto& stage : command.bindings.roots)
                for (auto& root : stage)
                    if (root.kind == 1)
                        root = {};
    });
    original<decltype(&heaps)>(28)(cmd, count, values);
}
template <unsigned PipelineDomain>
void STDMETHODCALLTYPE signature(ID3D12GraphicsCommandList* cmd, ID3D12RootSignature* value) {
    change(cmd, [&](Command& command) {
        if (command.bindings.signatures[PipelineDomain].Get() != value)
            command.bindings.roots[PipelineDomain] = {};
        command.bindings.signatures[PipelineDomain] = value;
    });
    original<decltype(&signature<PipelineDomain>)>(29 + PipelineDomain)(cmd, value);
}
template <unsigned PipelineDomain>
void STDMETHODCALLTYPE table(ID3D12GraphicsCommandList* cmd, UINT index,
                             D3D12_GPU_DESCRIPTOR_HANDLE value) {
    change(cmd, [&](Command& command) {
        check(index < 64, "phoenix root index");
        auto& root_binding = command.bindings.roots[PipelineDomain][index];
        root_binding = {};
        root_binding.kind = 1;
        root_binding.address = value.ptr;
    });
    original<decltype(&table<PipelineDomain>)>(31 + PipelineDomain)(cmd, index, value);
}
template <unsigned PipelineDomain>
void constants_value(ID3D12GraphicsCommandList* cmd, UINT index, UINT count, const void* values,
                     UINT offset) {
    change(cmd, [&](Command& command) {
        check(index < 64 && offset <= 64 && count <= 64 - offset, "phoenix root constant extent");
        auto& root_binding = command.bindings.roots[PipelineDomain][index];
        if (root_binding.kind != 2)
            root_binding = {};
        root_binding.kind = 2;
        const auto* data = static_cast<const uint32_t*>(values);
        for (UINT slot_index = 0; slot_index < count; ++slot_index) {
            root_binding.constants[offset + slot_index] = data[slot_index];
            root_binding.mask |= uint64_t(1) << (offset + slot_index);
        }
    });
}
template <unsigned PipelineDomain>
void STDMETHODCALLTYPE constant(ID3D12GraphicsCommandList* cmd, UINT index, UINT value,
                                UINT offset) {
    constants_value<PipelineDomain>(cmd, index, 1, &value, offset);
    original<decltype(&constant<PipelineDomain>)>(33 + PipelineDomain)(cmd, index, value, offset);
}
template <unsigned PipelineDomain>
void STDMETHODCALLTYPE constants(ID3D12GraphicsCommandList* cmd, UINT index, UINT count,
                                 const void* values, UINT offset) {
    constants_value<PipelineDomain>(cmd, index, count, values, offset);
    original<decltype(&constants<PipelineDomain>)>(35 + PipelineDomain)(cmd, index, count, values,
                                                                        offset);
}
template <unsigned PipelineDomain, unsigned Kind>
void STDMETHODCALLTYPE address(ID3D12GraphicsCommandList* cmd, UINT index,
                               D3D12_GPU_VIRTUAL_ADDRESS value) {
    change(cmd, [&](Command& command) {
        check(index < 64, "phoenix root address index");
        auto& root_binding = command.bindings.roots[PipelineDomain][index];
        root_binding = {};
        root_binding.kind = Kind;
        root_binding.address = value;
    });
    original<decltype(&address<PipelineDomain, Kind>)>(37 + PipelineDomain +
                                                       (Kind - 3) * 2)(cmd, index, value);
}
inline void STDMETHODCALLTYPE enhanced(ID3D12GraphicsCommandList7* cmd, UINT count,
                                       const D3D12_BARRIER_GROUP* groups) {
    change(cmd, [](Command& command) {
        mark(command, unsupported_enhanced_barrier);
        command.states.clear();
        command.depth_plane_states.clear();
    });
    original<decltype(&enhanced)>(80)(cmd, count, groups);
}
inline void STDMETHODCALLTYPE bundle(ID3D12GraphicsCommandList* cmd,
                                     ID3D12GraphicsCommandList* child) {
    change(cmd, [](Command& command) { mark(command, unsupported_execute_bundle); });
    original<decltype(&bundle)>(27)(cmd, child);
}
inline void STDMETHODCALLTYPE predication(ID3D12GraphicsCommandList* cmd, ID3D12Resource* buffer,
                                          UINT64 offset, D3D12_PREDICATION_OP predicate_operation) {
    change(cmd, [&](Command& command) { command.predicated = buffer != nullptr; });
    original<decltype(&predication)>(55)(cmd, buffer, offset, predicate_operation);
}
inline void STDMETHODCALLTYPE indirect(ID3D12GraphicsCommandList* cmd, ID3D12CommandSignature* sig,
                                       UINT max_count, ID3D12Resource* args, UINT64 offset,
                                       ID3D12Resource* counts, UINT64 count_offset) {
    change(cmd, [&](Command& command) {
        auto& observer_state = observer();
        const auto resource_state_entry =
            sig ? observer_state.signatures.find(sig) : observer_state.signatures.end();
        if (resource_state_entry == observer_state.signatures.end()) {
            ++command.indirect_unknown_signature;
            mark(command, unsupported_execute_indirect);
            return;
        }
        if (!resource_state_entry->second.neutral) {
            ++command.indirect_rejected_signature;
            command.indirect_rejected_type_mask |= resource_state_entry->second.type_mask;
            mark(command, unsupported_execute_indirect);
            return;
        }
        ++command.indirect_accepted;
    });
    original<decltype(&indirect)>(59)(cmd, sig, max_count, args, offset, counts, count_offset);
}
// ID3D12Device slot 41. Creation is the only place the argument descriptors
// are visible: by ExecuteIndirect time the signature is an opaque object and
// its arguments live in a GPU buffer.
inline HRESULT STDMETHODCALLTYPE create_command_signature(ID3D12Device* device,
                                                          const D3D12_COMMAND_SIGNATURE_DESC* desc,
                                                          ID3D12RootSignature* root, REFIID riid,
                                                          void** out) {
    const auto result =
        device_original<decltype(&create_command_signature)>(41)(device, desc, root, riid, out);
    if (FAILED(result) || !desc || !out || !*out)
        return result;
    try {
        SignatureInfo info;
        info.arguments = desc->NumArgumentDescs;
        info.byte_stride = desc->ByteStride;
        info.neutral = desc->NumArgumentDescs > 0 && desc->pArgumentDescs != nullptr;
        for (UINT slot_index = 0; slot_index < desc->NumArgumentDescs && desc->pArgumentDescs;
             ++slot_index) {
            const auto type = desc->pArgumentDescs[slot_index].Type;
            if (unsigned(type) < 31)
                info.type_mask |= 1u << unsigned(type);
            if (!binding_neutral(type))
                info.neutral = false;
        }
        // A signature carrying a root signature only needs one because some
        // argument writes root arguments, so it is never binding neutral.
        if (root)
            info.neutral = false;
        auto& observer_state = observer();
        std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
        if (observer_state.signatures.size() < 4096)
            observer_state.signatures[static_cast<ID3D12CommandSignature*>(*out)] = info;
    } catch (
        ...) { /* Classification is evidence. Failing to record it leaves the signature unknown, which refuses. */
    }
    return result;
}
// Forward: defined after forget_resource/erase below.
inline bool install_resource_release(ID3D12Resource* resource);
inline ULONG STDMETHODCALLTYPE command_release(ID3D12GraphicsCommandList* cmd);
// ID3D12Device slot 27. Only the plain CreateCommittedResource is hooked.
// CreateCommittedResource1/2/3 carry protected sessions and enhanced initial
// layouts that this contract does not model, so a resource made through them
// is simply never registered and therefore never answers.
inline HRESULT STDMETHODCALLTYPE create_committed_resource(
    ID3D12Device* device, const D3D12_HEAP_PROPERTIES* heap, D3D12_HEAP_FLAGS heap_flags,
    const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES initial, const D3D12_CLEAR_VALUE* clear,
    REFIID riid, void** out) {
    const auto result = device_original<decltype(&create_committed_resource)>(27)(
        device, heap, heap_flags, desc, initial, clear, riid, out);
    if (FAILED(result) || !out || !*out || !heap || !desc)
        return result;
    try {
        CommittedInfo info;
        info.initial_state = uint32_t(initial);
        // DEFAULT heap only: UPLOAD and READBACK have fixed states this
        // contract has no reason to touch, and CUSTOM is unmodelled.
        if (heap->Type != D3D12_HEAP_TYPE_DEFAULT)
            info.ineligible_mask |= ineligible_heap_type;
        // A shared or cross-adapter resource can be transitioned by a process
        // we do not observe at all.
        if (heap_flags & (D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER))
            info.ineligible_mask |= ineligible_shared_heap;
        if (desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
            info.ineligible_mask |= ineligible_not_texture2d;
        // Simultaneous access allows use from several queues at once, so a
        // single tracked state is not meaningful.
        if (desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)
            info.ineligible_mask |= ineligible_simultaneous_access;
        // Explicit UNORDERED_ACCESS only. COMMON would drag in promotion and
        // decay, which are not modelled, so they are refused rather than
        // guessed -- exactly the case the contract review called out.
        if (initial != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            info.ineligible_mask |= ineligible_initial_state;
        info.eligible = info.ineligible_mask == 0;
        auto& observer_state = observer();
        auto* resource = static_cast<ID3D12Resource*>(*out);
        // Registration requires a destruction feed. Either this resource's own
        // Release is observed (installed here, on first sight of its vtable),
        // or the host feeds destruction externally. Without one of the two the
        // record would linger past the object's death and a recycled pointer
        // could inherit a creation answer it never earned. Refusing to
        // register fails closed: the resource simply never answers. A hook
        // failure throws, which counts as unobserved rather than skipping
        // the external-feed fallback below.
        bool release_observed = false;
        try {
            release_observed = install_resource_release(resource);
        } catch (...) {
        }
        info.release_observed = release_observed;
        std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
        if (!release_observed && !observer_state.external_destruction_feed)
            return result;
        if (observer_state.committed.size() >= 262144 &&
            !observer_state.committed.count(resource)) {
            observer_state.committed_overflow = true;
            return result;
        }
        // A recycled pointer must never inherit the previous resource's
        // history, so this overwrites rather than merges.
        observer_state.committed[resource] = info;
    } catch (...) { /* An unrecorded creation simply never answers. */
    }
    return result;
}
inline void install_device(ID3D12Device* device, HookBatch& hooks) {
    if (!device)
        return;
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    auto* vtable = *reinterpret_cast<void***>(device);
    auto add = [&](unsigned slot, auto detour, const char* what) {
        auto* target = vtable[slot];
        if (observer_state.device_targets[slot]) {
            check(observer_state.device_targets[slot] == target,
                  "phoenix device implementation changed");
            return;
        }
        hooks.add(target, reinterpret_cast<void*>(detour), &observer_state.device_originals[slot],
                  &observer_state.device_targets[slot]);
    };
    add(27, create_committed_resource, "phoenix committed resource hook");
    add(41, create_command_signature, "phoenix command signature hook");
}
inline void install_device(ID3D12Device* device) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    HookBatch hooks;
    install_device(device, hooks);
    hooks.commit();
}
inline void STDMETHODCALLTYPE begin_pass(ID3D12GraphicsCommandList4* cmd, UINT count,
                                         const D3D12_RENDER_PASS_RENDER_TARGET_DESC* render_targets,
                                         const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC* depth,
                                         D3D12_RENDER_PASS_FLAGS flags) {
    change(cmd, [](Command& command) { command.render_pass = true; });
    original<decltype(&begin_pass)>(68)(cmd, count, render_targets, depth, flags);
}
inline void STDMETHODCALLTYPE end_pass(ID3D12GraphicsCommandList4* cmd) {
    change(cmd, [](Command& command) { command.render_pass = false; });
    original<decltype(&end_pass)>(69)(cmd);
}
inline void STDMETHODCALLTYPE state_object(ID3D12GraphicsCommandList4* cmd,
                                           ID3D12StateObject* state) {
    change(cmd, [](Command& command) { mark(command, unsupported_state_object); });
    original<decltype(&state_object)>(75)(cmd, state);
}
inline void STDMETHODCALLTYPE protected_session(ID3D12GraphicsCommandList3* cmd,
                                                ID3D12ProtectedResourceSession* session) {
    change(cmd, [&](Command& command) {
        if (session)
            mark(command, unsupported_protected_session);
    });
    original<decltype(&protected_session)>(67)(cmd, session);
}
inline void STDMETHODCALLTYPE meta(ID3D12GraphicsCommandList4* cmd, ID3D12MetaCommand* value,
                                   const void* parameters, SIZE_T size) {
    change(cmd, [](Command& command) { mark(command, unsupported_meta_command); });
    original<decltype(&meta)>(71)(cmd, value, parameters, size);
}
inline void install(ID3D12GraphicsCommandList* cmd, bool freshly_created = false) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    check(cmd && cmd->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT,
          "phoenix observer requires DIRECT command list");
    check(observer_state.commands.size() < 512 || observer_state.commands.count(cmd),
          "phoenix command observation limit");
    HookBatch hooks;
    auto* vtable = *reinterpret_cast<void***>(cmd);
    auto add = [&](unsigned slot, auto detour, void** table_ptr) {
        auto* target = table_ptr[slot];
        if (observer_state.targets[slot]) {
            check(observer_state.targets[slot] == target, "phoenix command implementation changed");
            return;
        }
        hooks.add(target, reinterpret_cast<void*>(detour), &observer_state.originals[slot],
                  &observer_state.targets[slot]);
    };
    add(2, command_release, vtable);
    add(9, close, vtable);
    add(10, reset, vtable);
    add(11, clear, vtable);
    add(25, pipeline, vtable);
    add(26, barriers, vtable);
    add(27, bundle, vtable);
    add(28, heaps, vtable);
    add(29, signature<0>, vtable);
    add(30, signature<1>, vtable);
    add(31, table<0>, vtable);
    add(32, table<1>, vtable);
    add(33, constant<0>, vtable);
    add(34, constant<1>, vtable);
    add(35, constants<0>, vtable);
    add(36, constants<1>, vtable);
    add(37, address<0, 3>, vtable);
    add(38, address<1, 3>, vtable);
    add(39, address<0, 4>, vtable);
    add(40, address<1, 4>, vtable);
    add(41, address<0, 5>, vtable);
    add(42, address<1, 5>, vtable);
    add(55, predication, vtable);
    add(59, indirect, vtable);
    ComPtr<ID3D12GraphicsCommandList4> cmd4;
    if (SUCCEEDED(cmd->QueryInterface(IID_PPV_ARGS(&cmd4)))) {
        check(cmd4.Get() == cmd, "phoenix nonidentical command interface pointer");
        auto* v4 = *reinterpret_cast<void***>(cmd4.Get());
        add(67, protected_session, v4);
        add(68, begin_pass, v4);
        add(69, end_pass, v4);
        add(71, meta, v4);
        add(75, state_object, v4);
    }
    ComPtr<ID3D12GraphicsCommandList7> cmd7;
    if (SUCCEEDED(cmd->QueryInterface(IID_PPV_ARGS(&cmd7)))) {
        check(cmd7.Get() == cmd, "phoenix nonidentical enhanced command pointer");
        add(80, enhanced, *reinterpret_cast<void***>(cmd7.Get()));
    }
    // Classify command signatures from here on. Anything the engine created
    // before this point stays unknown and therefore refused; that is reported
    // separately so "hooked too late" never reads as "rejected".
    ComPtr<ID3D12Device> device;
    if (SUCCEEDED(cmd->GetDevice(IID_PPV_ARGS(&device))))
        install_device(device.Get(), hooks);
    hooks.commit();
    auto entry = observer_state.commands.try_emplace(cmd);
    if (entry.second)
        entry.first->second.epoch_known = freshly_created;
}
inline void erase(ID3D12GraphicsCommandList* cmd) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    observer_state.commands.erase(cmd);
}
inline void forget_resource(ID3D12Resource* resource) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    for (auto& entry : observer_state.commands) {
        entry.second.states.erase(resource);
        entry.second.depth_plane_states.erase(resource);
        entry.second.discarded_states.erase(resource);
        entry.second.aliased.erase(resource);
        entry.second.deactivated.erase(resource);
    }
    observer_state.committed.erase(resource);
}
// IUnknown::Release is vtable slot 2 on every COM object. When the original
// returns 0 the object is dead: no record keyed by its pointer may survive,
// because the next allocation may recycle the address. The original is always
// called first and its value returned, so an untracked object pays one
// indirect call and one comparison and is otherwise untouched.
//
// Erasing here is idempotent with the ReShade host's destroy_resource event:
// whichever feed observes the death first retires the record and the other
// finds nothing to do. A lost race against a concurrent recreation at the
// same address can only refuse a live resource (fail closed), never answer
// for a dead one, because creation always overwrites the record afterwards.
template <unsigned HookSlot> ULONG STDMETHODCALLTYPE resource_release_at(ID3D12Resource* self) {
    using Fn = ULONG(__stdcall*)(ID3D12Resource*);
    const auto original = reinterpret_cast<Fn>(
        observer().resource_release_originals[HookSlot].load(std::memory_order_acquire));
    const ULONG refs = original(self);
    if (refs == 0 && self) {
        observer().resource_final_releases.fetch_add(1, std::memory_order_relaxed);
        forget_resource(self);
    }
    return refs;
}
inline ULONG STDMETHODCALLTYPE command_release(ID3D12GraphicsCommandList* cmd) {
    const auto result = original<decltype(&command_release)>(2)(cmd);
    if (result == 0 && cmd) {
        observer().command_final_releases.fetch_add(1, std::memory_order_relaxed);
        erase(cmd);
    }
    return result;
}
inline void* resource_release_detour(unsigned slot) noexcept {
    switch (slot) {
    case 0:
        return reinterpret_cast<void*>(&resource_release_at<0>);
    case 1:
        return reinterpret_cast<void*>(&resource_release_at<1>);
    case 2:
        return reinterpret_cast<void*>(&resource_release_at<2>);
    case 3:
        return reinterpret_cast<void*>(&resource_release_at<3>);
    case 4:
        return reinterpret_cast<void*>(&resource_release_at<4>);
    case 5:
        return reinterpret_cast<void*>(&resource_release_at<5>);
    case 6:
        return reinterpret_cast<void*>(&resource_release_at<6>);
    default:
        return reinterpret_cast<void*>(&resource_release_at<7>);
    }
}
// Hooks Release on a resource's implementation vtable, once per distinct
// vtable. Called from the creation detour, before the new resource is handed
// to anyone, so no Release on it can precede the hook. Returns false when the
// vtable cannot be observed (table full or hook failure), in which case the
// caller must not register the resource unless an external destruction feed
// covers it. Must not be called with the observer lock held: enabling a hook
// suspends other threads, which may be inside this observer.
inline bool install_resource_release(ID3D12Resource* resource) {
    if (!resource)
        return false;
    auto& observer_state = observer();
    auto* vtable = *reinterpret_cast<void***>(resource);
    if (!vtable || !vtable[2])
        return false;
    void* target = vtable[2];
    unsigned slot = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
        for (unsigned slot_index = 0; slot_index < observer_state.resource_release_hooked;
             ++slot_index)
            if (observer_state.resource_release_targets[slot_index] == target)
                return true;
        if (observer_state.resource_release_claimed >= Observer::kResourceReleaseVtables) {
            observer_state.resource_release_overflow = true;
            return false;
        }
        slot = observer_state.resource_release_claimed++;
    }
    auto init = MH_Initialize();
    check(init == MH_OK || init == MH_ERROR_ALREADY_INITIALIZED,
          "phoenix resource release hook initialization");
    void* original = nullptr;
    check(MH_CreateHook(target, resource_release_detour(slot), &original) == MH_OK,
          "phoenix resource release hook creation");
    // Stored before enabling: the detour reads only its own slot, so it must
    // never observe a live hook with a null original.
    observer_state.resource_release_originals[slot].store(original, std::memory_order_release);
    if (MH_EnableHook(target) != MH_OK) {
        // Release the created-but-disabled hook so a later retry starts
        // clean instead of failing with MH_ERROR_ALREADY_CREATED.
        MH_RemoveHook(target);
        check(false, "phoenix resource release hook enable");
    }
    {
        std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
        // Another thread may have covered this same target while our hook
        // was being enabled; ours is then redundant but harmless, because
        // MinHook chains hooks on the same target.
        for (unsigned slot_index = 0; slot_index < observer_state.resource_release_hooked;
             ++slot_index)
            if (observer_state.resource_release_targets[slot_index] == target)
                return true;
        observer_state.resource_release_targets[slot] = target;
        if (observer_state.resource_release_hooked < slot + 1)
            observer_state.resource_release_hooked = slot + 1;
        return true;
    }
}
inline Json reference_snapshot() {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    uint64_t heaps_count = 0, signatures = 0, pipelines = 0, states = 0, discarded = 0, closed = 0,
             depth = 0, planar_seen = 0, planar_filtered = 0, splits = 0, wildcards = 0;
    std::set<const void*> unique;
    for (const auto& entry : observer_state.commands) {
        const auto& command = entry.second;
        states += command.states.size();
        discarded += command.discarded_states.size();
        depth += command.depth_plane_states.size();
        planar_seen += command.planar_transitions_seen;
        planar_filtered += command.planar_transitions_filtered;
        splits += command.split_barriers;
        wildcards += command.wildcard_aliasing_barriers;
        if (command.closed)
            ++closed;
        for (const auto& heap : command.bindings.heaps)
            if (heap) {
                ++heaps_count;
                unique.insert(heap.Get());
            }
        for (const auto& sig : command.bindings.signatures)
            if (sig) {
                ++signatures;
                unique.insert(sig.Get());
            }
        if (command.bindings.pipeline) {
            ++pipelines;
            unique.insert(command.bindings.pipeline.Get());
        }
    }
    return {{"commands", observer_state.commands.size()},
            {"closed_commands", closed},
            {"epoch_state_entries", states},
            {"discarded_state_entries", discarded},
            {"depth_plane_entries", depth},
            {"planar_transitions_seen", planar_seen},
            {"planar_transitions_filtered", planar_filtered},
            {"split_barriers", splits},
            {"wildcard_aliasing_barriers", wildcards},
            {"strong_heap_reference_slots", heaps_count},
            {"strong_signature_reference_slots", signatures},
            {"strong_pipeline_reference_slots", pipelines},
            {"distinct_binding_object_pointers", unique.size()},
            {"state_maps_hold_resource_references", false}};
}
// `require_output` is false when the caller writes to its OWN output texture
// whose state it created and therefore already knows; the engine output state
// is then neither needed nor invented.
// Epoch-local state lookup for a single resource; empty when this recording
// has not established it.
inline bool epoch_state(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource,
                        D3D12_RESOURCE_STATES& out) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    const auto command_entry = observer_state.commands.find(cmd);
    if (command_entry == observer_state.commands.end() || !resource)
        return false;
    const auto found = command_entry->second.states.find(resource);
    if (found == command_entry->second.states.end())
        return false;
    out = found->second;
    return true;
}
// Independently observed depth plane; stencil state is intentionally separate.
inline bool depth_plane_state(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource,
                              D3D12_RESOURCE_STATES& out) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    const auto resource_state_entry = observer_state.commands.find(cmd);
    if (resource_state_entry == observer_state.commands.end() || !resource ||
        resource_state_entry->second.deactivated.count(resource))
        return false;
    const auto found = resource_state_entry->second.depth_plane_states.find(resource);
    if (found == resource_state_entry->second.depth_plane_states.end())
        return false;
    out = found->second;
    return true;
}
// Diagnosis only, never a resolve basis: the state this recording
// established for a resource before a wildcard aliasing barrier discarded
// it. Tells a refusal whether the engine transitions the output in this
// list at all (discarded present) or never does (discarded absent).
inline bool discarded_epoch_state(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource,
                                  D3D12_RESOURCE_STATES& out) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    const auto command_entry = observer_state.commands.find(cmd);
    if (command_entry == observer_state.commands.end() || !resource)
        return false;
    const auto found = command_entry->second.discarded_states.find(resource);
    if (found == command_entry->second.discarded_states.end())
        return false;
    out = found->second;
    return true;
}
// Observation completeness only: the epoch must be known from its Reset, still
// open, and free of any construct that poisoned the recording. Says nothing
// about aliasing; see aliasing_permits.
inline bool permits_ledger_fallback(ID3D12GraphicsCommandList* cmd) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    const auto resource_state_entry = observer_state.commands.find(cmd);
    return resource_state_entry != observer_state.commands.end() &&
           resource_state_entry->second.epoch_known && !resource_state_entry->second.closed &&
           !resource_state_entry->second.unsupported;
}
// Whether aliasing observed in this epoch permits an older ledger answer for
// one resource. Two separate facts:
//   * A NAMED operand is direct evidence that this resource was aliased over,
//     so nothing older than the barrier may be used for it. The bounded alias
//     operand log is diagnostic; the aliased set is the complete named set.
//   * A WILDCARD operand names no resource, so it is not evidence about this
//     one. An aliasing barrier redefines CONTENTS, not the state D3D12 tracks
//     per resource, and the state is the only thing this answer is used for:
//     the StateBefore of our own transition. Contents would matter if we read
//     the resource, and every resource we read is gated far more strictly, by
//     resolve(), which demands a transition in the current epoch. Measured:
//     a wildcard-only epoch was refused in sandbox a67 while builds without
//     this veto produced 512/512 and 2048/2048 byte-exact pairs against the
//     untouched DLL, so the veto costs correctness and buys nothing.
// Why aliasing_permits answered as it did, for the refusal record.
// Names the ExecuteIndirect situation on a list. `unknown` means a signature
// was used that we never saw created -- the engine built it before our device
// hook existed -- which is a coverage gap, not a rejected signature.
// `rejected_types` is a bitmask over D3D12_INDIRECT_ARGUMENT_TYPE, so a
// refusal says precisely which argument kinds a wider contract would have to
// model.
// Process-wide ExecuteIndirect evidence for the run summary. `device_hooked`
// false means no command signature can ever be classified, so every indirect
// refuses as unknown; that is a coverage statement, not a rejection.
inline Json indirect_summary() {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    uint64_t accepted = 0, unknown = 0, rejected = 0, neutral = 0;
    uint32_t mask = 0;
    for (const auto& entry : observer_state.commands) {
        accepted += entry.second.indirect_accepted;
        unknown += entry.second.indirect_unknown_signature;
        rejected += entry.second.indirect_rejected_signature;
        mask |= entry.second.indirect_rejected_type_mask;
    }
    for (const auto& entry : observer_state.signatures)
        if (entry.second.neutral)
            ++neutral;
    return {{"device_hooked", observer_state.device_targets[41] != nullptr},
            {"signatures_classified", observer_state.signatures.size()},
            {"signatures_binding_neutral", neutral},
            {"calls_accepted", accepted},
            {"calls_refused_unknown_signature", unknown},
            {"calls_refused_binding_signature", rejected},
            {"refused_argument_type_mask", mask}};
}
// The creation-state answer, and the only way this observer will ever produce
// one. It requires ALL of:
//   * we saw the resource created by the plain CreateCommittedResource, so it
//     owns its own implicit heap and nothing can alias its memory,
//   * it passed every eligibility rule at creation,
//   * no barrier on any command list has ever named it,
//   * the registry never overflowed, so an absent record really does mean
//     "not created here" rather than "evicted".
// A resource that fails any of these gets no answer at all.
inline bool committed_creation_state(ID3D12Resource* resource, D3D12_RESOURCE_STATES& out) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    if (!resource || observer_state.committed_overflow)
        return false;
    const auto resource_state_entry = observer_state.committed.find(resource);
    if (resource_state_entry == observer_state.committed.end())
        return false;
    if (!resource_state_entry->second.eligible ||
        resource_state_entry->second.touched_by_named_barrier)
        return false;
    out = D3D12_RESOURCE_STATES(resource_state_entry->second.initial_state);
    return true;
}
// The observed creation state WITHOUT the never-barriered requirement.
//
// `committed_creation_state` is the one a correctness answer may use; this one
// is not. It exists solely so a display feature can put something on screen in
// a game that never transitions its output, and every caller must treat the
// result as an assumption about the present rather than evidence.
inline bool committed_creation_state_unchecked(ID3D12Resource* resource,
                                               D3D12_RESOURCE_STATES& out) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    if (!resource || observer_state.committed_overflow)
        return false;
    const auto resource_state_entry = observer_state.committed.find(resource);
    if (resource_state_entry == observer_state.committed.end() ||
        !resource_state_entry->second.eligible)
        return false;
    out = D3D12_RESOURCE_STATES(resource_state_entry->second.initial_state);
    return true;
}
inline std::string committed_detail(ID3D12Resource* resource) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    const auto resource_state_entry =
        resource ? observer_state.committed.find(resource) : observer_state.committed.end();
    std::string text =
        "committed_registry=" + std::to_string(observer_state.committed.size()) +
        " overflow=" + std::string(observer_state.committed_overflow ? "yes" : "no") +
        " device_hooked=" + std::string(observer_state.device_targets[27] ? "yes" : "no");
    if (resource_state_entry == observer_state.committed.end())
        return text + " creation_observed=no";
    char mask[16];
    sprintf_s(mask, "%x", resource_state_entry->second.ineligible_mask);
    return text + " creation_observed=yes eligible=" +
           std::string(resource_state_entry->second.eligible ? "yes" : "no") +
           " ineligible_mask=0x" + mask + " touched_by_named_barrier=" +
           std::string(resource_state_entry->second.touched_by_named_barrier ? "yes" : "no") +
           " seen_wildcard_aliasing=" +
           std::string(resource_state_entry->second.seen_wildcard_aliasing ? "yes" : "no") +
           " release_observed=" +
           std::string(resource_state_entry->second.release_observed ? "yes" : "no") +
           " initial_state=" + std::to_string(resource_state_entry->second.initial_state);
}
inline Json committed_summary() {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    uint64_t eligible = 0, named = 0, wildcard = 0, answerable = 0;
    // Histogram over Ineligible bits, so a run says WHICH rule excluded the
    // resources rather than only that none qualified.
    uint64_t by_rule[6]{};
    for (const auto& resource_entry : observer_state.committed) {
        if (resource_entry.second.eligible)
            ++eligible;
        if (resource_entry.second.touched_by_named_barrier)
            ++named;
        if (resource_entry.second.seen_wildcard_aliasing)
            ++wildcard;
        if (resource_entry.second.eligible && !resource_entry.second.touched_by_named_barrier)
            ++answerable;
        for (unsigned bit = 0; bit < 6; ++bit)
            if (resource_entry.second.ineligible_mask & (1u << bit))
                ++by_rule[bit];
    }
    return {{"device_hooked", observer_state.device_targets[27] != nullptr},
            {"resources_observed", observer_state.committed.size()},
            {"eligible_shape", eligible},
            {"touched_by_named_barrier", named},
            {"seen_wildcard_aliasing", wildcard},
            {"answerable", answerable},
            {"excluded_heap_type", by_rule[0]},
            {"excluded_shared_heap", by_rule[1]},
            {"excluded_not_texture2d", by_rule[2]},
            {"excluded_simultaneous_access", by_rule[3]},
            {"excluded_initial_state", by_rule[4]},
            {"overflow", observer_state.committed_overflow},
            {"release_vtables_hooked", observer_state.resource_release_hooked},
            {"release_overflow", observer_state.resource_release_overflow},
            {"external_destruction_feed", observer_state.external_destruction_feed},
            {"resource_final_releases",
             observer_state.resource_final_releases.load(std::memory_order_relaxed)},
            {"command_final_releases",
             observer_state.command_final_releases.load(std::memory_order_relaxed)}};
}
inline std::string indirect_detail(ID3D12GraphicsCommandList* cmd) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    const auto resource_state_entry = observer_state.commands.find(cmd);
    if (resource_state_entry == observer_state.commands.end())
        return "command not observed";
    const auto& command = resource_state_entry->second;
    return "indirect_accepted=" + std::to_string(command.indirect_accepted) +
           " unknown_signature=" + std::to_string(command.indirect_unknown_signature) +
           " rejected_signature=" + std::to_string(command.indirect_rejected_signature) +
           " rejected_types=0x" +
           [&] {
               char hex_buffer[16];
               sprintf_s(hex_buffer, "%x", command.indirect_rejected_type_mask);
               return std::string(hex_buffer);
           }() +
           " signatures_classified=" + std::to_string(observer_state.signatures.size()) +
           " device_hooked=" + std::string(observer_state.device_targets[41] ? "yes" : "no");
}
inline std::string aliasing_detail(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    const auto resource_state_entry = observer_state.commands.find(cmd);
    if (resource_state_entry == observer_state.commands.end())
        return "command not observed";
    const auto& command = resource_state_entry->second;
    std::string text = "aliasing_barriers=" + std::to_string(command.aliasing_barriers) +
                       " wildcard=" + std::to_string(command.wildcard_aliasing_barriers) +
                       " named_resources=" + std::to_string(command.aliased.size()) +
                       " resource_named=" +
                       std::string(resource && command.aliased.count(resource) ? "yes" : "no") +
                       " epoch_states=" + std::to_string(command.states.size()) +
                       " discarded=" + std::to_string(command.discarded_states.size());
    return text;
}
inline bool aliasing_permits(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    const auto resource_state_entry = observer_state.commands.find(cmd);
    if (resource_state_entry == observer_state.commands.end())
        return false;
    return !(resource && resource_state_entry->second.aliased.count(resource));
}
// Only for resources just consumed/produced by a successful engine SR/RR
// call on THIS list. Never use this to resurrect an older ledger state.
// Aliasing erases epoch states; a fresh transition must reestablish them.
// A last named-before operand remains refused until explicitly reactivated.
inline bool current_provider_resource(const Command& command, ID3D12Resource* resource) {
    return resource && command.states.count(resource) && !command.deactivated.count(resource);
}
inline bool current_provider_resource(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    auto resource_state_entry = observer_state.commands.find(cmd);
    return resource_state_entry != observer_state.commands.end() &&
           current_provider_resource(resource_state_entry->second, resource);
}
// A submission-boundary snapshot runs before the entire provider list. It is
// eligible only if neither input was transitioned/aliased earlier in that
// list before Evaluate. Read-only entry states are checked at submission too.
inline bool inputs_precede_recording(ID3D12GraphicsCommandList* cmd,
                                     ID3D12Resource* motion, ID3D12Resource* depth) {
    auto& source = observer();
    std::lock_guard<std::recursive_mutex> lock(source.mutex);
    const auto found = source.commands.find(cmd);
    if (found == source.commands.end()) return false;
    const auto& recording = found->second;
    if (!recording.epoch_known || recording.closed || recording.unsupported ||
        recording.render_pass || recording.predicated || recording.aliasing_barriers) return false;
    for (auto* input : {motion,depth})
        if (!input || recording.states.count(input) || recording.depth_plane_states.count(input) ||
            recording.discarded_states.count(input) || recording.deactivated.count(input)) return false;
    return true;
}
inline std::string command_refusal_detail(ID3D12GraphicsCommandList* cmd) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    auto resource_state_entry = observer_state.commands.find(cmd);
    if (resource_state_entry == observer_state.commands.end())
        return "unobserved";
    const auto& command = resource_state_entry->second;
    return "mask=" + std::to_string(command.unsupported_mask) +
           " epoch=" + std::to_string(command.epoch_known) +
           " heaps=" + std::to_string(command.bindings.heaps_known) +
           " pass=" + std::to_string(command.render_pass) +
           " predicated=" + std::to_string(command.predicated) +
           " closed=" + std::to_string(command.closed);
}
inline std::shared_ptr<Bindings> snapshot_bindings(ID3D12GraphicsCommandList* cmd) {
    auto& source = observer();
    std::lock_guard<std::recursive_mutex> lock(source.mutex);
    const auto found = source.commands.find(cmd);
    check(found != source.commands.end(), "phoenix command not observed from creation/reset");
    const auto& command = found->second;
    check(command.epoch_known && !command.unsupported && !command.render_pass &&
              !command.predicated && !command.closed && command.bindings.heaps_known,
          "phoenix unknown/unsupported command bindings or barrier path");
    return std::allocate_shared<Bindings>(
        host_memory::Allocator<Bindings, host_memory::Tag::snapshots>{}, command.bindings);
}
inline std::shared_ptr<Bindings> resolve(ID3D12GraphicsCommandList* cmd,
                                         const NVSDK_NGX_Parameter* params, FrameEvidence& frame,
                                         bool require_output = true) {
    auto& observer_state = observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    const auto command_entry = observer_state.commands.find(cmd);
    check(command_entry != observer_state.commands.end(),
          "phoenix command not observed from creation/reset");
    const auto& command = command_entry->second;
    check(command.epoch_known && !command.unsupported && !command.render_pass &&
              !command.predicated && !command.closed && command.bindings.heaps_known,
          "phoenix unknown/unsupported command bindings or barrier path");
    auto state = [&](const char* key, bool optional) {
        auto* resource = parameter_resource(params, key);
        if (optional && !resource)
            return D3D12_RESOURCE_STATE_COMMON;
        const auto found = command.states.find(resource);
        if (!resource || found == command.states.end())
            throw std::runtime_error(
                std::string("phoenix state not observed in current recording epoch: ") + key);
        return found->second;
    };
    frame.color_state = state("DLSSNR.Color", false);
    frame.motion_state = state("DLSSNR.MVec", true);
    frame.output_state =
        require_output ? state("DLSSNR.Output", false) : D3D12_RESOURCE_STATE_COMMON;
    frame.states_verified = true;
    frame.output_state_verified = require_output;
    auto snapshot = snapshot_bindings(cmd);
    frame.restore_bindings = [](void* context, ID3D12GraphicsCommandList* command) {
        static_cast<Bindings*>(context)->restore(command);
    };
    frame.restore_context = snapshot.get();
    return snapshot;
}
} // namespace production_game41::observation
