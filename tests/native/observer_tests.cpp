#include "../../src/native/support/observer.h"
#include <iostream>
using namespace phoenix_addon;
namespace obs = phoenix_addon::observation;

// CPU COM-ABI fakes. No device, queue, NVAPI library or native DLL is created.
struct Object {
    void** vtable;
    ULONG refs = 1;
};
struct FakeCommand : Object {
    unsigned calls = 0;
    uint64_t last = 0;
    HRESULT result = S_OK;
};
struct FakeResource : Object {
    D3D12_RESOURCE_DESC desc{};
};
struct FakeParameters : NVSDK_NGX_Parameter {
    std::map<std::string, ID3D12Resource*> resources;
    void Set(const char*, unsigned long long) override {}
    void Set(const char*, float) override {}
    void Set(const char*, double) override {}
    void Set(const char*, unsigned) override {}
    void Set(const char*, int) override {}
    void Set(const char*, ID3D11Resource*) override {}
    void Set(const char*, ID3D12Resource*) override {}
    void Set(const char*, void*) override {}
    NVSDK_NGX_Result Get(const char*, unsigned long long*) const override {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    NVSDK_NGX_Result Get(const char*, float*) const override {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    NVSDK_NGX_Result Get(const char*, double*) const override {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    NVSDK_NGX_Result Get(const char*, unsigned*) const override {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    NVSDK_NGX_Result Get(const char*, int*) const override {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    NVSDK_NGX_Result Get(const char*, ID3D11Resource**) const override {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    NVSDK_NGX_Result Get(const char*, ID3D12Resource**) const override {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    NVSDK_NGX_Result Get(const char* key, void** value) const override {
        const auto lookup_iterator = resources.find(key);
        *value = lookup_iterator == resources.end() ? nullptr : lookup_iterator->second;
        return lookup_iterator == resources.end() ? NVSDK_NGX_Result_FAIL_InvalidParameter
                                                  : NVSDK_NGX_Result_Success;
    }
    void Reset() override {}
};
static ULONG STDMETHODCALLTYPE add_ref(IUnknown* ptr) {
    return ++reinterpret_cast<Object*>(ptr)->refs;
}
static ULONG STDMETHODCALLTYPE release_ref(IUnknown* ptr) {
    return --reinterpret_cast<Object*>(ptr)->refs;
}
static void STDMETHODCALLTYPE heaps(ID3D12GraphicsCommandList* cmd, UINT count,
                                    ID3D12DescriptorHeap* const*) {
    auto& fake_command = *reinterpret_cast<FakeCommand*>(cmd);
    ++fake_command.calls;
    fake_command.last = count;
}
static void STDMETHODCALLTYPE pointer(ID3D12GraphicsCommandList* cmd, void* value) {
    auto& fake_command = *reinterpret_cast<FakeCommand*>(cmd);
    ++fake_command.calls;
    fake_command.last = uint64_t(value);
}
static void STDMETHODCALLTYPE root(ID3D12GraphicsCommandList* cmd, UINT index, UINT64 value) {
    auto& fake_command = *reinterpret_cast<FakeCommand*>(cmd);
    ++fake_command.calls;
    fake_command.last = value + index;
}
static void STDMETHODCALLTYPE constant(ID3D12GraphicsCommandList* cmd, UINT, UINT value,
                                       UINT offset) {
    auto& fake_command = *reinterpret_cast<FakeCommand*>(cmd);
    ++fake_command.calls;
    fake_command.last = value + offset;
}
static void STDMETHODCALLTYPE constants(ID3D12GraphicsCommandList*, UINT, UINT, const void*, UINT) {
}
static void STDMETHODCALLTYPE barriers(ID3D12GraphicsCommandList*, UINT,
                                       const D3D12_RESOURCE_BARRIER*) {}
static void STDMETHODCALLTYPE enhanced(ID3D12GraphicsCommandList7*, UINT,
                                       const D3D12_BARRIER_GROUP*) {}
static void STDMETHODCALLTYPE execute_indirect(ID3D12GraphicsCommandList* cmd,
                                               ID3D12CommandSignature*, UINT, ID3D12Resource*,
                                               UINT64, ID3D12Resource*, UINT64) {
    ++reinterpret_cast<FakeCommand*>(cmd)->calls;
}
// Stands in for the real ID3D12Device slot 41 so the classifier can be driven
// without a device: it hands back the signature object the test supplies.
static Object* pending_signature = nullptr;
static Object* pending_resource = nullptr;
static HRESULT STDMETHODCALLTYPE create_committed(ID3D12Device*, const D3D12_HEAP_PROPERTIES*,
                                                  D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC*,
                                                  D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*,
                                                  REFIID, void** out) {
    *out = pending_resource;
    return pending_resource ? S_OK : E_FAIL;
}
static HRESULT STDMETHODCALLTYPE create_signature(ID3D12Device*,
                                                  const D3D12_COMMAND_SIGNATURE_DESC*,
                                                  ID3D12RootSignature*, REFIID, void** out) {
    *out = pending_signature;
    return pending_signature ? S_OK : E_FAIL;
}
static HRESULT STDMETHODCALLTYPE close(ID3D12GraphicsCommandList* cmd) {
    return reinterpret_cast<FakeCommand*>(cmd)->result;
}
static HRESULT STDMETHODCALLTYPE reset(ID3D12GraphicsCommandList* cmd, ID3D12CommandAllocator*,
                                       ID3D12PipelineState*) {
    return reinterpret_cast<FakeCommand*>(cmd)->result;
}
// SDK ID3D12ResourceVtbl explicitly specifies the Windows struct-return ABI.
static D3D12_RESOURCE_DESC* STDMETHODCALLTYPE description(ID3D12Resource* resource,
                                                          D3D12_RESOURCE_DESC* result) {
    *result = reinterpret_cast<FakeResource*>(resource)->desc;
    return result;
}
// MinHook is stubbed, not linked: these tests drive the Release-hook install
// path with injectable failures instead of patching real code.
static MH_STATUS g_mh_create_result = MH_OK;
static MH_STATUS g_mh_enable_result = MH_OK;
static unsigned g_mh_remove_calls = 0;
static unsigned g_mh_queue_calls = 0, g_mh_apply_calls = 0;
static MH_STATUS g_mh_queue_result = MH_OK, g_mh_apply_result = MH_OK;
MH_STATUS WINAPI MH_Initialize(void) {
    return MH_OK;
}
MH_STATUS WINAPI MH_CreateHook(LPVOID target, LPVOID, LPVOID* original) {
    if (g_mh_create_result != MH_OK)
        return g_mh_create_result;
    *original = target;
    return MH_OK;
}
MH_STATUS WINAPI MH_EnableHook(LPVOID) {
    return g_mh_enable_result;
}
MH_STATUS WINAPI MH_QueueEnableHook(LPVOID) {
    ++g_mh_queue_calls;
    return g_mh_queue_result;
}
MH_STATUS WINAPI MH_ApplyQueued() {
    ++g_mh_apply_calls;
    return g_mh_apply_result;
}
MH_STATUS WINAPI MH_RemoveHook(LPVOID) {
    ++g_mh_remove_calls;
    return MH_OK;
}
static unsigned cases = 0;
template <class F> void rejected(F action) {
    bool caught = false;
    try {
        action();
    } catch (const std::exception&) {
        caught = true;
    }
    check(caught, "observer negative case accepted");
    ++cases;
}
int main() {
    try {
        {
            void *first = nullptr, *first_original = nullptr, *second = nullptr, *second_original = nullptr;
            const auto target = reinterpret_cast<void*>(uintptr_t(0x1000));
            const auto other = reinterpret_cast<void*>(uintptr_t(0x2000));
            {
                HookBatch hooks;
                hooks.add(target, other, &first_original, &first);
                hooks.add(other, target, &second_original, &second);
                check(g_mh_queue_calls == 2 && g_mh_apply_calls == 0, "Hooks activated before batch commit");
                hooks.commit();
                check(g_mh_apply_calls == 1, "Hook activation did not use one batch");
            }
            check(first == target && second == other && g_mh_remove_calls == 0, "Committed hooks were rolled back");
            first = first_original = second = second_original = nullptr;
            {
                HookBatch hooks;
                hooks.add(target, other, &first_original, &first);
                hooks.add(other, target, &second_original, &second);
            }
            check(!first && !first_original && !second && !second_original && g_mh_remove_calls == 2,
                  "Uncommitted hooks retained stale target/trampoline pointers");
            g_mh_queue_result = MH_ERROR_MEMORY_PROTECT;
            rejected([&] { HookBatch hooks; hooks.add(target, other, &first_original, &first); });
            check(!first && !first_original && g_mh_remove_calls == 3, "Queue failure did not roll back its created hook");
            g_mh_queue_result = MH_OK;
            g_mh_apply_result = MH_ERROR_MEMORY_PROTECT;
            rejected([&] { HookBatch hooks; hooks.add(target, other, &first_original, &first); hooks.commit(); });
            check(!first && !first_original && g_mh_remove_calls == 4, "Activation failure retained a half-installed hook");
            g_mh_apply_result = MH_OK;
            const auto applies = g_mh_apply_calls;
            { HookBatch empty; empty.commit(); }
            check(g_mh_apply_calls == applies, "Empty hook batch enumerated threads");
            g_mh_remove_calls = 0;
            cases += 6;
        }
        std::array<void*, 81> vt{};
        vt[1] = reinterpret_cast<void*>(add_ref);
        vt[2] = reinterpret_cast<void*>(release_ref);
        vt[9] = reinterpret_cast<void*>(close);
        vt[10] = reinterpret_cast<void*>(reset);
        vt[25] = reinterpret_cast<void*>(pointer);
        vt[26] = reinterpret_cast<void*>(barriers);
        vt[28] = reinterpret_cast<void*>(heaps);
        vt[29] = vt[30] = reinterpret_cast<void*>(pointer);
        vt[31] = vt[32] = reinterpret_cast<void*>(root);
        vt[33] = vt[34] = reinterpret_cast<void*>(constant);
        vt[35] = vt[36] = reinterpret_cast<void*>(constants);
        for (unsigned iteration_index = 37; iteration_index <= 42; ++iteration_index)
            vt[iteration_index] = reinterpret_cast<void*>(root);
        vt[80] = reinterpret_cast<void*>(enhanced);
        FakeCommand fake_command{{vt.data()}};
        auto* cmd = reinterpret_cast<ID3D12GraphicsCommandList*>(&fake_command);
        Object heap{vt.data()}, signature{vt.data()}, pipeline{vt.data()};
        auto reset_vtable = vt;
        reset_vtable[10] = reinterpret_cast<void*>(description);
        FakeResource color{{reset_vtable.data()}}, output{{reset_vtable.data()}};
        color.desc.Dimension = output.desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        color.desc.MipLevels = output.desc.MipLevels = 1;
        color.desc.DepthOrArraySize = output.desc.DepthOrArraySize = 1;
        FakeParameters parameters;
        parameters.resources["DLSSNR.Color"] = reinterpret_cast<ID3D12Resource*>(&color);
        parameters.resources["DLSSNR.Output"] = reinterpret_cast<ID3D12Resource*>(&output);
        const NVSDK_NGX_Parameter* params = &parameters;
        using NativeGet = NVSDK_NGX_Result (*)(const NVSDK_NGX_Parameter*, const char*, void**);
        const auto native_get =
            reinterpret_cast<NativeGet>((*reinterpret_cast<void***>(&parameters))[8]);
        void* value = nullptr;
        check(NVSDK_NGX_SUCCEED(native_get(params, "DLSSNR.Color", &value)) && value == &color,
              "native resource getter vtable +0x40 ABI");
        ++cases;
        check(parameter_resource(params, "DLSSNR.Color", true) ==
                  reinterpret_cast<ID3D12Resource*>(&color),
              "resolver must use native void** getter, not typed D3D12 getter");
        ++cases;
        auto& observer_state = obs::observer();
        observer_state.originals = vt;
        observer_state.commands.try_emplace(cmd);
        FrameEvidence frame;
        rejected([&] { obs::resolve(cmd, params, frame); });
        obs::reset(cmd, nullptr, reinterpret_cast<ID3D12PipelineState*>(&pipeline));
        check(obs::permits_ledger_fallback(cmd), "known clean epoch permits ledger lookup");
        ++cases;
        auto* descriptor_heap = reinterpret_cast<ID3D12DescriptorHeap*>(&heap);
        obs::heaps(cmd, 1, &descriptor_heap);
        obs::signature<0>(cmd, reinterpret_cast<ID3D12RootSignature*>(&signature));
        obs::table<0>(cmd, 3, {12345});
        obs::heaps(cmd, 1, &descriptor_heap);
        check(observer_state.commands.at(cmd).bindings.roots[0][3].address == 12345,
              "same heap must preserve root table");
        ++cases;
        obs::constant<0>(cmd, 4, 99, 1);
        const uint32_t words[]{11, 22};
        obs::constants<0>(cmd, 4, 2, words, 3);
        obs::address<1, 4>(cmd, 2, 0x123456780ULL);
        check(observer_state.commands.at(cmd).bindings.roots[0][4].mask ==
                  ((1ull << 1) | (1ull << 3) | (1ull << 4)),
              "partial root constants");
        ++cases;
        rejected([&] { obs::resolve(cmd, params, frame); });
        D3D12_RESOURCE_BARRIER transitions[2]{};
        transitions[0].Type = transitions[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        transitions[0].Transition = {
            parameters.resources.at("DLSSNR.Color"), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
        transitions[1].Transition = {parameters.resources.at("DLSSNR.Output"), 0,
                                     D3D12_RESOURCE_STATE_COMMON,
                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
        obs::barriers(cmd, 2, transitions);
        auto snapshot = obs::resolve(cmd, params, frame);
        check(frame.states_verified &&
                  frame.color_state == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE &&
                  frame.output_state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
              "observed engine states");
        ++cases;
        check(snapshot->roots[0][3].address == 12345, "snapshot root table");
        ++cases;
        obs::heaps(cmd, 0, nullptr);
        check(observer_state.commands.at(cmd).bindings.roots[0][3].kind == 0 &&
                  snapshot->roots[0][3].address == 12345,
              "heap change invalidates tracker but preserves snapshot");
        ++cases;
        const auto before = fake_command.calls;
        snapshot->restore(cmd);
        check(fake_command.calls > before + 4, "restore did not issue heap/pipeline/root bindings");
        ++cases;
        obs::close(cmd);
        rejected([&] { obs::resolve(cmd, params, frame); });
        check(!observer_state.commands.at(cmd).bindings.pipeline &&
                  !observer_state.commands.at(cmd).bindings.signatures[0] &&
                  observer_state.commands.at(cmd).bindings.heaps.empty() && snapshot->pipeline &&
                  snapshot->signatures[0] && !snapshot->heaps.empty(),
              "Close releases observer objects while independent snapshot stays valid");
        ++cases;
        obs::reset(cmd, nullptr, nullptr);
        check(observer_state.commands.at(cmd).epoch_known &&
                  observer_state.commands.at(cmd).states.empty(),
              "Reset must forget preceding-epoch resource states");
        ++cases;
        // A fully named aliasing barrier affects exactly the two resources it
        // names. It is not an unsupported construct, and it must not leave a
        // usable state behind for either named resource.
        obs::barriers(cmd, 2, transitions);
        D3D12_RESOURCE_BARRIER named{};
        named.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        named.Aliasing.pResourceBefore = parameters.resources.at("DLSSNR.Color");
        named.Aliasing.pResourceAfter = reinterpret_cast<ID3D12Resource*>(&signature);
        obs::barriers(cmd, 1, &named);
        check(obs::permits_ledger_fallback(cmd),
              "a named aliasing barrier does not poison epoch observation");
        ++cases;
        check(!obs::aliasing_permits(cmd, parameters.resources.at("DLSSNR.Color")),
              "a named operand must block any older ledger answer for that resource");
        ++cases;
        check(!obs::aliasing_permits(cmd, reinterpret_cast<ID3D12Resource*>(&signature)),
              "both named operands must be blocked, not just the first");
        ++cases;
        check(obs::aliasing_permits(cmd, parameters.resources.at("DLSSNR.Output")),
              "a named aliasing barrier must not block resources it did not name");
        ++cases;
        check(!observer_state.commands.at(cmd).unsupported,
              "named aliasing barrier is a supported construct");
        ++cases;
        check(observer_state.commands.at(cmd).states.count(
                  parameters.resources.at("DLSSNR.Color")) == 0 &&
                  observer_state.commands.at(cmd).states.count(
                      parameters.resources.at("DLSSNR.Output")) == 1,
              "named aliasing must forget only the resources it names");
        ++cases;
        rejected([&] { obs::resolve(cmd, params, frame); });
        // A wildcard operand names no resource, so every tracked state is
        // dropped; only transitions observed afterwards may be used again.
        obs::reset(cmd, nullptr, nullptr);
        obs::heaps(cmd, 1, &descriptor_heap);
        obs::barriers(cmd, 2, transitions);
        D3D12_RESOURCE_BARRIER wildcard{};
        wildcard.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        wildcard.Aliasing.pResourceBefore = parameters.resources.at("DLSSNR.Color");
        obs::barriers(cmd, 1, &wildcard);
        check(!observer_state.commands.at(cmd).unsupported &&
                  observer_state.commands.at(cmd).states.empty() &&
                  observer_state.commands.at(cmd).wildcard_aliasing_barriers == 1,
              "wildcard aliasing must drop every tracked state without inventing one");
        ++cases;
        // A wildcard names no resource. A queue-ordered submitted transition
        // remains admissible; only an untransitioned creation state is refused.
        check(obs::aliasing_permits(cmd, parameters.resources.at("DLSSNR.Output")),
              "wildcard aliasing must not veto a submitted-transition answer");
        ++cases;
        check(obs::aliasing_permits(cmd, parameters.resources.at("DLSSNR.Output")),
              "a wildcard names no resource and must not veto an unnamed one");
        ++cases;
        check(!obs::aliasing_permits(cmd, parameters.resources.at("DLSSNR.Color")),
              "an earlier named operand stays blocked across a later wildcard");
        ++cases;
        rejected([&] { obs::resolve(cmd, params, frame); });
        obs::barriers(cmd, 2, transitions);
        frame = {};
        auto after_alias = obs::resolve(cmd, params, frame);
        check(bool(after_alias) && frame.states_verified && frame.output_state_verified &&
                  frame.color_state == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE &&
                  frame.output_state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
              "a transition after a wildcard aliasing barrier re-establishes state");
        ++cases;
        // The discarded lookup reports the pre-wildcard state for diagnosis
        // without ever resolving it: epoch empty, discarded set, null-safe.
        D3D12_RESOURCE_STATES discarded{};
        check(!obs::discarded_epoch_state(cmd, nullptr, discarded) &&
                  !obs::discarded_epoch_state(nullptr, parameters.resources.at("DLSSNR.Output"),
                                              discarded),
              "discarded lookup is null-safe");
        ++cases;
        obs::barriers(cmd, 1, &wildcard);
        check(!obs::epoch_state(cmd, parameters.resources.at("DLSSNR.Output"), discarded) &&
                  obs::discarded_epoch_state(cmd, parameters.resources.at("DLSSNR.Output"),
                                             discarded) &&
                  discarded == D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
              "wildcard moves established state to discarded, never resolves");
        ++cases;
        observer_state.commands.at(cmd).discarded_states[parameters.resources.at("DLSSNR.Color")] =
            D3D12_RESOURCE_STATE_COMMON;
        obs::forget_resource(parameters.resources.at("DLSSNR.Color"));
        check(!observer_state.commands.at(cmd).states.count(
                  parameters.resources.at("DLSSNR.Color")) &&
                  !observer_state.commands.at(cmd).discarded_states.count(
                      parameters.resources.at("DLSSNR.Color")),
              "destroyed resource must not retain epoch or discarded metadata");
        ++cases;
        rejected([&] { obs::resolve(cmd, params, frame); });
        const auto refs = obs::reference_snapshot();
        check(refs["commands"] == 1 && refs["state_maps_hold_resource_references"] == false,
              "observer reference inventory must identify non-owning resource metadata");
        ++cases;
        // Inputs-only resolution is for callers writing to their own output.
        obs::reset(cmd, nullptr, nullptr);
        obs::heaps(cmd, 1, &descriptor_heap);
        D3D12_RESOURCE_BARRIER color_only = transitions[0];
        obs::barriers(cmd, 1, &color_only);
        rejected([&] {
            FrameEvidence engine;
            obs::resolve(cmd, params, engine, true);
        });
        frame = {};
        auto inputs_only = obs::resolve(cmd, params, frame, false);
        check(bool(inputs_only) && frame.states_verified && !frame.output_state_verified &&
                  frame.color_state == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
              "inputs-only resolution must not claim an engine output state");
        ++cases;
        obs::reset(cmd, nullptr, nullptr);
        obs::reset(cmd, nullptr, nullptr);
        obs::enhanced(reinterpret_cast<ID3D12GraphicsCommandList7*>(cmd), 0, nullptr);
        check(observer_state.commands.at(cmd).unsupported, "enhanced barriers must fail closed");
        ++cases;
        obs::reset(cmd, nullptr, nullptr);
        obs::barriers(cmd, 2, transitions);
        obs::barriers(cmd, 2, transitions);
        check(observer_state.commands.at(cmd).unsupported,
              "contradictory StateBefore must fail closed");
        ++cases;
        obs::reset(cmd, nullptr, nullptr);
        transitions[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
        obs::barriers(cmd, 1, transitions);
        check(observer_state.commands.at(cmd).unsupported, "split barrier must fail closed");
        ++cases;
        obs::reset(cmd, nullptr, nullptr);
        obs::constant<0>(cmd, 0, 1, 64);
        check(observer_state.commands.at(cmd).unsupported,
              "bad root offset must poison observer without escaping COM");
        ++cases;
        fake_command.result = E_FAIL;
        obs::reset(cmd, nullptr, nullptr);
        check(observer_state.commands.at(cmd).unsupported, "failed Reset must remain poisoned");
        ++cases;
        // Two earlier cases left shared fixtures poisoned on purpose: Close and
        // Reset return E_FAIL, and transitions[0] carries BEGIN_ONLY. Both
        // would fail the lists below for reasons unrelated to ExecuteIndirect.
        fake_command.result = S_OK;
        transitions[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        // ExecuteIndirect. A signature is classified at creation, because that
        // is the only place its argument descriptors exist; by dispatch time
        // the arguments are in a GPU buffer. Only whole draw/dispatch
        // arguments write no binding, so only those may be accepted.
        check(obs::binding_neutral(D3D12_INDIRECT_ARGUMENT_TYPE_DRAW) &&
                  obs::binding_neutral(D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED) &&
                  obs::binding_neutral(D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH) &&
                  obs::binding_neutral(D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_RAYS) &&
                  obs::binding_neutral(D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH),
              "whole draw/dispatch arguments write no binding");
        ++cases;
        check(!obs::binding_neutral(D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW) &&
                  !obs::binding_neutral(D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW) &&
                  !obs::binding_neutral(D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT) &&
                  !obs::binding_neutral(D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW) &&
                  !obs::binding_neutral(D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW) &&
                  !obs::binding_neutral(D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW),
              "every binding-writing argument type must be refused");
        ++cases;
        vt[59] = reinterpret_cast<void*>(execute_indirect);
        observer_state.originals = vt;
        observer_state.device_originals[41] = reinterpret_cast<void*>(create_signature);
        Object draw_sig{vt.data()}, mixed_sig{vt.data()}, rooted_sig{vt.data()},
            stale_sig{vt.data()};
        auto classify = [&](Object& object, std::vector<D3D12_INDIRECT_ARGUMENT_TYPE> types,
                            bool rooted) {
            std::vector<D3D12_INDIRECT_ARGUMENT_DESC> descs(types.size());
            for (size_t iteration_index = 0; iteration_index < types.size(); ++iteration_index)
                descs[iteration_index].Type = types[iteration_index];
            D3D12_COMMAND_SIGNATURE_DESC desc{};
            desc.ByteStride = 32;
            desc.NumArgumentDescs = UINT(descs.size());
            desc.pArgumentDescs = descs.data();
            pending_signature = &object;
            void* produced = nullptr;
            obs::create_command_signature(
                nullptr, &desc,
                rooted ? reinterpret_cast<ID3D12RootSignature*>(&signature) : nullptr,
                __uuidof(ID3D12CommandSignature), &produced);
            pending_signature = nullptr;
            return reinterpret_cast<ID3D12CommandSignature*>(&object);
        };
        auto* draw_only = classify(draw_sig, {D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH}, false);
        auto* mixed = classify(
            mixed_sig,
            {D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT, D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH}, false);
        auto* rooted = classify(rooted_sig, {D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH}, true);
        check(observer_state.signatures.at(draw_only).neutral &&
                  !observer_state.signatures.at(mixed).neutral &&
                  !observer_state.signatures.at(rooted).neutral,
              "classification must accept only binding-neutral signatures");
        ++cases;
        check(observer_state.signatures.at(mixed).type_mask ==
                  ((1u << unsigned(D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT)) |
                   (1u << unsigned(D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH))),
              "classification must record every argument type it saw");
        ++cases;
        obs::reset(cmd, nullptr, nullptr);
        obs::indirect(cmd, reinterpret_cast<ID3D12CommandSignature*>(&stale_sig), 1, nullptr, 0,
                      nullptr, 0);
        check(observer_state.commands.at(cmd).unsupported &&
                  observer_state.commands.at(cmd).indirect_unknown_signature == 1 &&
                  observer_state.commands.at(cmd).indirect_rejected_signature == 0,
              "a signature created before the hook is unknown, not rejected");
        ++cases;
        obs::reset(cmd, nullptr, nullptr);
        obs::indirect(cmd, mixed, 1, nullptr, 0, nullptr, 0);
        check(observer_state.commands.at(cmd).unsupported &&
                  observer_state.commands.at(cmd).indirect_rejected_signature == 1 &&
                  (observer_state.commands.at(cmd).indirect_rejected_type_mask &
                   (1u << unsigned(D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT))),
              "a binding-writing signature must be refused and named");
        ++cases;
        obs::reset(cmd, nullptr, nullptr);
        obs::indirect(cmd, rooted, 1, nullptr, 0, nullptr, 0);
        check(observer_state.commands.at(cmd).unsupported,
              "a signature carrying a root signature must be refused");
        ++cases;
        obs::reset(cmd, nullptr, nullptr);
        const auto before_calls = fake_command.calls;
        obs::indirect(cmd, draw_only, 1, nullptr, 0, nullptr, 0);
        check(!observer_state.commands.at(cmd).unsupported &&
                  observer_state.commands.at(cmd).indirect_accepted == 1 &&
                  fake_command.calls == before_calls + 1,
              "a binding-neutral indirect must be accepted and still forwarded");
        ++cases;
        check(obs::permits_ledger_fallback(cmd),
              "an accepted indirect must leave the epoch completely observed");
        ++cases;
        obs::reset(cmd, nullptr, reinterpret_cast<ID3D12PipelineState*>(&pipeline));
        obs::heaps(cmd, 1, &descriptor_heap);
        obs::signature<0>(cmd, reinterpret_cast<ID3D12RootSignature*>(&signature));
        obs::barriers(cmd, 2, transitions);
        obs::indirect(cmd, draw_only, 1, nullptr, 0, nullptr, 0);
        {
            FrameEvidence indirect_frame;
            auto claimed = obs::resolve(cmd, params, indirect_frame);
            check(bool(claimed) && indirect_frame.states_verified &&
                      observer_state.commands.at(cmd).indirect_accepted == 1,
                  "a list carrying only binding-neutral indirects stays claimable");
            ++cases;
            claimed.reset();
        }
        obs::reset(cmd, nullptr, nullptr);
        obs::indirect(cmd, draw_only, 1, nullptr, 0, nullptr, 0);
        obs::indirect(cmd, mixed, 1, nullptr, 0, nullptr, 0);
        check(observer_state.commands.at(cmd).unsupported &&
                  observer_state.commands.at(cmd).indirect_accepted == 1 &&
                  observer_state.commands.at(cmd).indirect_rejected_signature == 1,
              "one refused signature poisons a list that also had accepted ones");
        ++cases;
        check(obs::indirect_detail(cmd).find("rejected_signature=1") != std::string::npos,
              "the indirect diagnostic must name the refusal");
        ++cases;
        obs::reset(cmd, nullptr, nullptr);
        // Observed private committed texture. ReShade's init_resource never
        // reaches us for a device made before ReShade loaded, so this is the
        // only route to a creation state for such a resource -- and it is
        // deliberately narrow.
        observer_state.device_originals[27] = reinterpret_cast<void*>(create_committed);
        auto make = [&](Object& object, D3D12_HEAP_TYPE heap_type, D3D12_HEAP_FLAGS heap_flags,
                        D3D12_RESOURCE_DIMENSION dim, D3D12_RESOURCE_FLAGS flags,
                        D3D12_RESOURCE_STATES initial) {
            D3D12_HEAP_PROPERTIES heap_properties{};
            heap_properties.Type = heap_type;
            D3D12_RESOURCE_DESC resource_description{};
            resource_description.Dimension = dim;
            resource_description.Flags = flags;
            pending_resource = &object;
            void* produced = nullptr;
            obs::create_committed_resource(nullptr, &heap_properties, heap_flags,
                                           &resource_description, initial, nullptr,
                                           __uuidof(ID3D12Resource), &produced);
            pending_resource = nullptr;
            return reinterpret_cast<ID3D12Resource*>(&object);
        };
        Object good{reset_vtable.data()}, upload{reset_vtable.data()},
            shared_heap{reset_vtable.data()}, buffer_res{reset_vtable.data()},
            simultaneous{reset_vtable.data()}, common_state{reset_vtable.data()};
        auto* eligible = make(
            good, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        D3D12_RESOURCE_STATES creation_state{};
        check(obs::committed_creation_state(eligible, creation_state) &&
                  creation_state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
              "a private committed UAV texture answers its creation state");
        ++cases;
        struct Case {
            Object* object;
            D3D12_HEAP_TYPE heap;
            D3D12_HEAP_FLAGS flags;
            D3D12_RESOURCE_DIMENSION dim;
            D3D12_RESOURCE_FLAGS rflags;
            D3D12_RESOURCE_STATES state;
        };
        const Case refused[]{
            {&upload, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE,
             D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS},
            {&shared_heap, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED,
             D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS},
            {&buffer_res, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE,
             D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS},
            {&simultaneous, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE,
             D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS},
            {&common_state, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE,
             D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_COMMON},
        };
        for (const auto& refused_case : refused) {
            auto* resource = make(*refused_case.object, refused_case.heap, refused_case.flags,
                                  refused_case.dim, refused_case.rflags, refused_case.state);
            D3D12_RESOURCE_STATES ignored{};
            check(!obs::committed_creation_state(resource, ignored),
                  "ineligible committed resource must not answer");
            ++cases;
        }
        check(obs::committed_detail(eligible).find("eligible=yes") != std::string::npos &&
                  obs::committed_detail(reinterpret_cast<ID3D12Resource*>(&upload))
                          .find("ineligible_mask=0x1") != std::string::npos,
              "the committed diagnostic must name the rule that failed");
        ++cases;
        // A barrier naming it anywhere, on any list, ends the answer.
        D3D12_RESOURCE_BARRIER touch{};
        touch.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        touch.Transition = {eligible, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_COPY_SOURCE};
        obs::mark_barrier_touched(1, &touch);
        check(!obs::committed_creation_state(eligible, creation_state),
              "a transition anywhere ends the creation answer");
        ++cases;
        // A wildcard aliasing barrier ends every creation answer at once.
        Object fresh{reset_vtable.data()};
        auto* second =
            make(fresh, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE,
                 D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        check(obs::committed_creation_state(second, creation_state),
              "a fresh creation answers again");
        ++cases;
        D3D12_RESOURCE_BARRIER wildcard_alias_any{};
        wildcard_alias_any.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        wildcard_alias_any.Aliasing = {nullptr, nullptr};
        obs::mark_barrier_touched(1, &wildcard_alias_any);
        check(obs::committed_creation_state(second, creation_state) &&
                  creation_state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
              "a wildcard aliasing barrier must NOT end a private committed answer");
        ++cases;
        check(obs::committed_detail(second).find("seen_wildcard_aliasing=yes") != std::string::npos,
              "the wildcard must still be recorded even though it does not disqualify");
        ++cases;
        // A NAMED aliasing operand is different: it is positive evidence that
        // THAT resource was aliased over.
        D3D12_RESOURCE_BARRIER named_alias{};
        named_alias.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        named_alias.Aliasing = {nullptr, second};
        obs::mark_barrier_touched(1, &named_alias);
        check(!obs::committed_creation_state(second, creation_state),
              "a named aliasing operand does end the creation answer");
        ++cases;
        // Recreating at the same pointer must not inherit the old history.
        auto* again =
            make(fresh, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE,
                 D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        check(obs::committed_creation_state(again, creation_state),
              "a recycled pointer starts a fresh creation record");
        ++cases;
        obs::forget_resource(again);
        check(!obs::committed_creation_state(again, creation_state),
              "a destroyed resource must stop answering");
        ++cases;
        observer_state.committed_overflow = true;
        auto* blocked = make(
            good, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        check(!obs::committed_creation_state(blocked, creation_state),
              "registry overflow must disable the creation answer entirely");
        ++cases;
        observer_state.committed_overflow = false;
        observer_state.committed.clear();
        // Release-observed destruction. The ASI host has no destroy_resource
        // event, so a resource's own Release is the feed that retires its
        // record; without it a recycled pointer could inherit a creation
        // answer it never earned.
        observer_state.resource_release_originals[0].store(reinterpret_cast<void*>(release_ref),
                                                           std::memory_order_release);
        FakeResource dying{{reset_vtable.data()}};
        dying.desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        dying.desc.MipLevels = 1;
        dying.desc.DepthOrArraySize = 1;
        auto* mortal =
            make(dying, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE,
                 D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        check(observer_state.committed.count(mortal) == 1,
              "a release-observable creation registers");
        ++cases;
        check(obs::committed_detail(mortal).find("release_observed=yes") != std::string::npos,
              "the diagnostic must say the record dies with the object");
        ++cases;
        check(observer_state.resource_release_hooked == 1,
              "first sight of a vtable hooks its Release");
        ++cases;
        auto* mortal_again =
            make(dying, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE,
                 D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        check(observer_state.resource_release_hooked == 1 &&
                  observer_state.resource_release_claimed == 1,
              "a repeat sight of the same vtable reuses the hook");
        ++cases;
        (void)mortal_again;
        dying.refs = 2;
        check(obs::resource_release_at<0>(mortal) == 1, "a non-final Release forwards the count");
        ++cases;
        check(obs::committed_creation_state(mortal, creation_state),
              "a live resource keeps answering");
        ++cases;
        D3D12_RESOURCE_BARRIER mortal_barrier{};
        mortal_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        mortal_barrier.Transition = {mortal, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                     D3D12_RESOURCE_STATE_COPY_SOURCE};
        obs::barriers(cmd, 1, &mortal_barrier);
        check(observer_state.commands.at(cmd).states.count(mortal) == 1,
              "epoch state recorded for the scrub test");
        ++cases;
        dying.refs = 1;
        check(obs::resource_release_at<0>(mortal) == 0, "a final Release forwards zero");
        ++cases;
        check(observer_state.committed.count(mortal) == 0,
              "a released resource leaves no record behind");
        ++cases;
        check(observer_state.commands.at(cmd).states.count(mortal) == 0,
              "a final Release scrubs epoch states");
        ++cases;
        check(observer_state.resource_final_releases.load(std::memory_order_relaxed) == 1,
              "final Releases are counted");
        ++cases;
        // A full Release table fails closed: the resource is usable by the
        // game (its creation succeeded) but never answers.
        auto vt2 = vt;
        vt2[2] = reinterpret_cast<void*>(add_ref); // a vtable never seen before
        auto rv2 = vt2;
        rv2[10] = reinterpret_cast<void*>(description);
        Object late{rv2.data()};
        observer_state.resource_release_claimed = 8;
        check(!obs::install_resource_release(reinterpret_cast<ID3D12Resource*>(&late)),
              "a full Release table refuses the install");
        ++cases;
        check(observer_state.resource_release_overflow, "the refusal is recorded");
        ++cases;
        observer_state.resource_release_claimed = 1;
        observer_state.resource_release_overflow = false;
        // Registration gating: without any destruction feed the resource must
        // not register; an external feed (the ReShade host) covers it.
        g_mh_create_result = MH_ERROR_NOT_EXECUTABLE;
        observer_state.external_destruction_feed = false;
        Object strict{rv2.data()};
        auto* unobserved =
            make(strict, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE,
                 D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        check(observer_state.committed.count(unobserved) == 0,
              "without any destruction feed nothing registers");
        ++cases;
        observer_state.external_destruction_feed = true;
        Object covered{rv2.data()};
        auto* kept =
            make(covered, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE,
                 D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        check(obs::committed_creation_state(kept, creation_state),
              "an external destruction feed covers an unhookable vtable");
        ++cases;
        check(obs::committed_detail(kept).find("release_observed=no") != std::string::npos,
              "the diagnostic must say which feed covers the record");
        ++cases;
        g_mh_create_result = MH_OK;
        observer_state.external_destruction_feed = false;
        observer_state.committed.clear();
        // A hook that creates but never enables is removed, so a later retry
        // starts clean instead of failing with MH_ERROR_ALREADY_CREATED.
        g_mh_enable_result = MH_ERROR_MEMORY_PROTECT;
        observer_state.resource_release_claimed = 1;
        observer_state.resource_release_hooked = 1;
        Object retry{rv2.data()};
        auto* unwatched =
            make(retry, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE,
                 D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        check(observer_state.committed.count(unwatched) == 0,
              "a hook that never enables registers nothing");
        ++cases;
        check(g_mh_remove_calls == 1, "the disabled hook is removed for a clean retry");
        ++cases;
        g_mh_enable_result = MH_OK;
        observer_state.committed.clear();
        // The command list's own Release retires its record the same way, so
        // a recycled list pointer under the ASI host cannot inherit epoch
        // states from a dead recording.
        fake_command.refs = 2;
        check(obs::command_release(cmd) == 1, "a non-final list Release forwards the count");
        ++cases;
        check(observer_state.commands.count(cmd) == 1, "a live list keeps its record");
        ++cases;
        fake_command.refs = 1;
        check(obs::command_release(cmd) == 0, "a final list Release forwards zero");
        ++cases;
        check(observer_state.commands.count(cmd) == 0, "a released list is forgotten");
        ++cases;
        check(observer_state.command_final_releases.load(std::memory_order_relaxed) == 1,
              "final list Releases are counted");
        ++cases;
        obs::erase(cmd);
        snapshot.reset();
        after_alias.reset();
        inputs_only.reset();
        check(!obs::permits_ledger_fallback(cmd), "unobserved list cannot use ledger fallback");
        ++cases;
        check(!obs::aliasing_permits(cmd, nullptr),
              "unobserved list has no aliasing evidence either");
        ++cases;
        check(heap.refs == 1 && signature.refs == 1 && pipeline.refs == 1,
              "snapshot/tracker COM leases leaked");
        ++cases;
        {
            obs::Command provider;
            auto* resource = reinterpret_cast<ID3D12Resource*>(0x9870);
            provider.aliased.insert(resource);
            check(!obs::current_provider_resource(provider, resource),
                  "Historical alias without fresh transition refused");
            ++cases;
            provider.states[resource] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            check(obs::current_provider_resource(provider, resource),
                  "Fresh successful-provider resource admitted");
            ++cases;
            provider.deactivated.insert(resource);
            check(!obs::current_provider_resource(provider, resource),
                  "Explicitly deactivated provider resource refused");
            ++cases;
            provider.deactivated.erase(resource);
            provider.states.clear();
            check(!obs::current_provider_resource(provider, resource),
                  "Wildcard invalidation requires another transition");
            ++cases;
        }
        {
            observer_state.commands.try_emplace(cmd);
            obs::reset(cmd, nullptr, nullptr);
            FakeResource depth{{reset_vtable.data()}};
            depth.desc = color.desc;
            depth.desc.Format = DXGI_FORMAT_R32G8X24_TYPELESS;
            auto* resource = reinterpret_cast<ID3D12Resource*>(&depth);
            D3D12_RESOURCE_BARRIER transition{};
            transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            transition.Transition = {resource, 0, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                     D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
            obs::barriers(cmd, 1, &transition);
            D3D12_RESOURCE_STATES actual{};
            check(obs::depth_plane_state(cmd, resource, actual) &&
                      actual == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                  "Depth plane state missing");
            ++cases;
            transition.Transition = {resource, 1, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                     D3D12_RESOURCE_STATE_DEPTH_READ};
            obs::barriers(cmd, 1, &transition);
            check(obs::depth_plane_state(cmd, resource, actual) &&
                      actual == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                  "Stencil erased depth state");
            ++cases;
            check(!obs::epoch_state(cmd, resource, actual),
                  "Per-plane state leaked into whole-resource proof");
            ++cases;
            obs::barriers(cmd, 2, transitions);
            transition.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
            transition.Aliasing = {nullptr, nullptr};
            obs::barriers(cmd, 1, &transition);
            check(obs::depth_plane_state(cmd, resource, actual) &&
                      actual == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                  "Wildcard alias must preserve depth plane proof");
            ++cases;
            check(observer_state.commands.at(cmd).states.empty(),
                  "Wildcard alias still drops whole-resource epoch states");
            ++cases;
            check(observer_state.commands.at(cmd).wildcard_aliasing_barriers == 1,
                  "Wildcard alias counted");
            ++cases;
            FakeResource other{{reset_vtable.data()}};
            other.desc = color.desc;
            auto* other_texture = reinterpret_cast<ID3D12Resource*>(&other);
            D3D12_RESOURCE_BARRIER split{};
            split.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            split.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
            split.Transition = {other_texture, 0, D3D12_RESOURCE_STATE_COMMON,
                                D3D12_RESOURCE_STATE_COPY_SOURCE};
            obs::barriers(cmd, 1, &split);
            check(observer_state.commands.at(cmd).unsupported, "Split barrier still fails closed");
            ++cases;
            check(obs::depth_plane_state(cmd, resource, actual),
                  "Split on another resource must preserve depth proof");
            ++cases;
            split.Transition = {resource, 0, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                D3D12_RESOURCE_STATE_COPY_SOURCE};
            obs::barriers(cmd, 1, &split);
            check(!obs::depth_plane_state(cmd, resource, actual),
                  "Split on depth itself must erase its record");
            ++cases;
            obs::reset(cmd, nullptr, nullptr);
            FakeResource array{{reset_vtable.data()}};
            array.desc = color.desc;
            array.desc.Format = DXGI_FORMAT_R32G8X24_TYPELESS;
            array.desc.DepthOrArraySize = 2;
            auto* array_texture = reinterpret_cast<ID3D12Resource*>(&array);
            D3D12_RESOURCE_BARRIER shaped{};
            shaped.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            shaped.Transition = {array_texture, 0, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
            obs::barriers(cmd, 1, &shaped);
            check(!obs::depth_plane_state(cmd, array_texture, actual),
                  "Filtered planar depth must not resolve");
            ++cases;
            check(observer_state.commands.at(cmd).planar_transitions_filtered == 1,
                  "Filtered planar transition counted");
            ++cases;
            check(observer_state.commands.at(cmd).planar_transitions_seen == 0,
                  "Shape-filtered depth is not a sighting");
            ++cases;
            obs::reset(cmd, nullptr, nullptr);
            obs::observer().commands.erase(cmd);
        }
        check(!GetModuleHandleW(L"nvapi64.dll") && !GetModuleHandleW(L"nvngx_dlssnr.dll"),
              "CPU tests loaded a GPU/native library");
        ++cases;
        std::cout << "{\"cpu_only\":true,\"passed\":true,\"cases\":" << cases
                  << ",\"nvapi_loaded\":false,\"native_nr_loaded\":false}\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
