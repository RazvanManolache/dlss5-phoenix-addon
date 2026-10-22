// ReShade addon entry point: overlay, settings, game callbacks and runtime routing.
// The native RTX and portable C++ routes share UI controls but keep separate execution/session ownership.

// imgui.h must precede reshade.hpp: ReShade routes ImGui through its own
// instance (REFERENCE.md "Overlays"), so no imgui sources are built.
// ImTextureID=ImU64 comes from the build; the headers must match the
// running ReShade's imgui version (SDK-paired).
#define IMGUI_DISABLE_INCLUDE_IMCONFIG_H
#include <imgui.h>
#include <reshade.hpp>
#include <atomic>
#include "comparison_hud.h"
#include "../../pipeline/recovery.h"
#include "../../pipeline/screenshot_receipt.h"
#include "../compatibility/controller.h"
#define PRODUCTION_GAME_PHOENIX_EXPORTS
#include "../../native/session/session.h"
#include "../../native/support/observer.h"
#include "../../native/support/ledger.h"
#include "../../native/support/timing.h"
#include "../../native/support/gpu_timing.h"
#include "../../native/support/worker_timing.h"
#include "../../native/support/worker_budget.h"
#include "../../native/support/routing.h"
#include "../../native/support/panel_visibility.h"
#include "../../native/support/comparison.h"
#include "../../native/support/overrides.h"
#include "../../native/support/control_bridge.h"
#include "../../native/support/direct.h"
#include "feeder_parameters.h"
#include "../../native/support/present_guides.h"
#include "../../native/support/offload.h"
#include "../../native/support/device_catalog.h"
#include "../../native/support/model_status.h"
#include "../../native/support/model_tools.h"
#include "../../native/support/model_selection_state.h"
#include "../../native/support/optional_feature_lifetime.h"
#include "../../native/support/diagnostic.h"
#include <dxgi1_4.h>
#include <psapi.h>
#include "../../../third_party/minhook/include/MinHook.h"
#include <intrin.h>
namespace {
phoenix_addon::phoenix::direct::AppearanceTextures harvest_appearance(
    ID3D12GraphicsCommandList*,const NVSDK_NGX_Parameter*,phoenix_addon::FrameEvidence&);
}
#include "../../native/support/ngx_shim.h"
#include "../../native/support/menu_helper.h"
#include "../../../third_party/streamline/sl.h"
#include "../../../third_party/streamline/sl_dlss.h"
#include "../../../third_party/streamline/sl_dlss_d.h"
#include "../../native/support/sl_shim.h"
#include <cstdio>
#include <set>
#include <optional>
#ifndef CNR_PHOENIX_DIAGNOSTIC
#define CNR_PHOENIX_DIAGNOSTIC 0
#endif
#if CNR_PHOENIX_DIAGNOSTIC
#include "../../native/support/diagnostic.h"
extern "C" __declspec(dllexport) const char* NAME = "Phoenix DLSS5 diagnostic comparison";
extern "C" __declspec(dllexport) const char* DESCRIPTION =
    "Bounded paired inference with independent histories; not a production replacement or benchmark.";
#else

#if defined(PHOENIX_EXPERIMENTAL_NATIVE)
extern "C" __declspec(dllexport) const char* NAME = "Phoenix DLSS5 experimental RTX compatibility";
#else
extern "C" __declspec(dllexport) const char* NAME = "Phoenix DLSS5 reconstructed NR";
#endif
extern "C" __declspec(dllexport) const char* DESCRIPTION =
    "Explicit-contract candidate NR integration; no native fallback for registered features.";
#endif

namespace {
using namespace phoenix_addon;
static_assert(ngx_shim::kSuperSampling == static_cast<unsigned>(NVSDK_NGX_Feature_SuperSampling),
              "Direct-game SR routing must match the NGX SDK feature ID");
thread_local unsigned phoenix_ngx_depth = 0;
thread_local bool phoenix_render_stage = false, phoenix_present_stage = false;
thread_local int phoenix_present_encoding = 2;
using Evaluate = NVSDK_NGX_Result (*)(void*, ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*,
                                      const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);
using Execute = void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);
using Release = NVSDK_NGX_Result (*)(const NVSDK_NGX_Handle*);
using Create = NVSDK_NGX_Result (*)(void*, ID3D12GraphicsCommandList*, const NVSDK_NGX_Parameter*,
                                    NVSDK_NGX_Handle**);
struct State {
    std::recursive_mutex mutex;
    phoenix::Settings panel;
    std::vector<phoenix::ExecutionDevice> execution_devices;
    std::filesystem::path worker_python, worker_script, worker_weights, worker_vulkan_build,
        worker_hip_build, worker_sycl_build;
    PhoenixLiveControls& live = panel.controls;
    std::array<phoenix::Settings, 4> presets;
    std::array<int, 6> hotkeys{{VK_F9, VK_F10, VK_F11, VK_F8, VK_F7, 0}};
    // Offline input-capture burst bookkeeping: base path + frames written.
    std::string capture_base;
    unsigned captures_taken = 0;
    uint64_t hotkey_deadline = 0;
    bool timing_settings_dirty = true;
    int capture_hotkey = -1, last_mode = -1;
    int reset_hold = 0;
    unsigned last_transformer_blocks = 8;
    std::array<uint64_t, 9> depth_frames{};
    phoenix::DepthBudget depth_budget;
    uint64_t fps_tick = 0, fps_presents = 0;
    double present_fps = 0;
    uint64_t budget_recorded = 0, budget_record_tick = 0;
    uint64_t routed_frames = 0, switches = 0, markers = 0;
    unsigned routing_test_period = 0;
    std::array<uint64_t, 4> mode_frames{}, mode_own{}, mode_native{};
    Json route_events = Json::array();
    phoenix::Settings displayed;
    uint64_t display_tick = 0, override_tick = 0, override_updates = 0;
    uint64_t hud_draws = 0;
    std::array<uint64_t, 6> comparison_frames{};
    std::filesystem::file_time_type override_mtime{};
    bool override_seen = false;
    std::string override_status = "Not loaded";
    // Staged NVIDIA diagnosis: 0 = nothing, 1 = owned barriers only,
    // 2 = +dispatch, 3 = +engine copy, 4 = +native NR, 5 = +resolve.
    int screenshot_request_seen = 0;
    struct OutputLease {
        std::shared_ptr<Session> session;
        ComPtr<ID3D12Resource> output;
    };
    std::vector<OutputLease> route_outputs;
    struct ComparisonLease {
        std::shared_ptr<Session> session;
        std::unique_ptr<phoenix::Comparison> compositor;
    };
    std::vector<ComparisonLease> comparisons;
    std::vector<std::shared_ptr<diagnostic::Pair>> route_pairs;
    std::filesystem::path route_capture;
    bool live_loaded = false, interactive = false, observe = false;
    uint64_t live_resets_recorded = 0;
    Endpoint endpoint;
    Evaluate original_evaluate = nullptr;
    Execute original_execute = nullptr;
    Release original_release = nullptr;
    Create original_create = nullptr;
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) original_dispatch = nullptr;
    dispatch_audit::Counts dispatch_counts;
    host_memory::Set<uint64_t, host_memory::Tag::fresh_features> fresh_features;
    bool fresh_claim = false;
    bool cold_restart = false;
    void* evaluate_target = nullptr;
    // NGX export interposition (renodx replacement, stage A). One slot per
    // known NGX copy; observers only, never claims. Originals are stored
    // typed; each detour forwards first, then records.
    using NgxCreateFn = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*, NVSDK_NGX_Feature,
                                                      NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
    using NgxEvaluateFn = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*,
                                                        const NVSDK_NGX_Handle*,
                                                        const NVSDK_NGX_Parameter*,
                                                        PFN_NVSDK_NGX_ProgressCallback);
    using NgxReleaseFn = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);
    struct NgxCopy {
        HMODULE module = nullptr;
        NgxCreateFn create = nullptr;
        NgxEvaluateFn evaluate = nullptr;
        NgxReleaseFn release = nullptr;
        bool hooked = false;
    };
    std::array<NgxCopy, 3> ngx_copies;
    ngx_shim::Observer ngx_observed;
    uint64_t ngx_export_hooks = 0, ngx_features_seen = 0;
    // Streamline tag harvest (renodx replacement, sl stage). renodx runs
    // NGX-hooks-only, so sl.interposer is solely ours: no order games, no
    // caller-identity hazard. Observers only, same discipline as above.
    ::PFun_slEvaluateFeature* sl_evaluate = nullptr;
    ::PFun_slSetTagForFrame* sl_settag = nullptr;
    ::PFun_slSetTag* sl_settag_old = nullptr;
    // Self-hosted native NR (renodx replacement, direct stage): our own NR
    // feature evaluated piggyback on the claimed game SR/RR frame with its
    // real color and motion. Feature 18 goes through nvngx_dlssnr.dll's own
    // exports (probe-verified: Init_Ext plus CreateFeature plus Evaluate,
    // gated on our "nvngx.dll" module substring). Feature 13 (Ray
    // Reconstruction, an unavailable separate experiment) goes through the NGX core instead,
    // normally on the game's own core init. Active only with no foreign
    // driver present; failures only withhold the NVIDIA source, never the
    // DLSS5 Phoenix frame.
    struct SelfDrive {
        NVSDK_NGX_Handle* handle = nullptr;
        NVSDK_NGX_Parameter* params = nullptr;
        State::NgxCreateFn create_fn = nullptr;
        State::NgxEvaluateFn evaluate_fn = nullptr;
        State::NgxReleaseFn release_fn = nullptr;
        ComPtr<ID3D12Resource> output;
        struct Refinement { NVSDK_NGX_Handle* handle = nullptr; ComPtr<ID3D12Resource> output; };
        std::vector<Refinement> refinement;
        ID3D12Resource* result() const { return refinement.empty() ? output.Get() : refinement.back().output.Get(); }
        ComPtr<ID3D12Device> device;
        HMODULE nr_module = nullptr, core_module = nullptr, forward_module = nullptr;
        bool inited = false, nr_initialized = false, core_init_attempted = false, req_queried = false;
        unsigned feature_id = 0, in_w = 0, in_h = 0, attempts = 0;
        int requested_feature = 0;
        unsigned warmup = 0;
        bool failed = false, first = true;
        int bridge = -1, hdr_mode = -1, auto_exp = -1;
        unsigned preset = 0, passes = 1;
        std::string source_choice, requested_contract;
        uint64_t creates = 0, evaluates = 0;
        uint64_t depth_present = 0, depth_proven = 0, depth_typed = 0;
        unsigned long long last_create_result = 0, last_evaluate_result = 0;
        unsigned long long retry_after_tick = 0;
        unsigned abandons = 0;
        unsigned long long dll_bytes = 0;
    };
    SelfDrive selfdrive;
    std::string model_switch_error;
    uint64_t nvidia_retirements = 0;
    bool feeder_nr_initialization_recorded = false;
    bool direct_game = false;
    bool direct_feeder = false;
    std::optional<uint32_t> feeder_stream;
    uint64_t feeder_calls = 0;
    std::array<std::unique_ptr<phoenix::direct::Runner>, 3> feeder_parked;
    std::array<std::unique_ptr<phoenix::GpuTiming>, 3> feeder_parked_timing;
    std::array<unsigned, 3> feeder_last_blocks{{8, 8, 8}};
    std::array<bool, 3> feeder_reset_pending{{true, true, true}};
    std::array<uint64_t, 3> feeder_stream_recorded{}, feeder_stream_resets{};
    bool feeder_applied_reset = false;
    // Self-hosted NR liveness for the current frame (panel gating) and the
    // last NR status line. Updated on every direct frame that requests NR.
    bool direct_nr_ready = false;
    std::string direct_nr_status;
    bool direct_srgb = false;
    std::unique_ptr<phoenix::direct::Runner> direct_runner;
    std::unique_ptr<phoenix::direct::PresentGuides> present_guides;
    std::unique_ptr<phoenix_compat::D3D12Bridge> present_bridge;
    reshade::api::effect_runtime* present_runtime = nullptr;
    uint64_t present_serial = 0;
    std::string direct_status = "Waiting for game DLSS evaluation";
    uint64_t direct_calls = 0, direct_deferred = 0, direct_busy = 0;
    uint64_t direct_status_tick = 0;
    Json bridge_inputs = nullptr;
    Json bridge_pending_response = nullptr;
    uint64_t bridge_input_tick = 0, bridge_poll_tick = 0;
    std::string bridge_session;
    std::string bridge_last_id, bridge_status;
    bool bridge_endpoint_written = false;
    bool direct_capture = false, direct_capture_done = false;
    unsigned capture_every = 0;
    unsigned long long next_capture_at = 0;
    unsigned direct_capture_frame = 1200;
    HMODULE sl_module = nullptr;
    bool sl_hooked = false;
    sl_shim::SlObserver sl_observed;
    uint64_t sl_export_hooks = 0, sl_tags_seen = 0, sl_evaluates_seen = 0;
    void* execute_target = nullptr;
    HMODULE self = nullptr;
    std::optional<uint64_t> claimed_feature;
    host_memory::Vector<std::shared_ptr<Session>, host_memory::Tag::retired> retired;
    host_memory::Vector<reshade::api::command_queue*, host_memory::Tag::queues> queues;
    host_memory::Map<ID3D12GraphicsCommandList*, std::shared_ptr<Session>,
                     host_memory::Tag::submitted>
        submitted_lists;
    // Which queue an engine command list is actually submitted on, observed
    // from real ExecuteCommandLists calls. Never inferred from a queue list.
    struct Ownership {
        ID3D12CommandQueue* queue = nullptr;
        uint64_t submissions = 0;
        bool ambiguous = false;
    };
    host_memory::Map<ID3D12GraphicsCommandList*, Ownership, host_memory::Tag::ownership> ownership;
    // Supplied (possibly wrapped) Evaluate list -> the original it wraps.
    host_memory::Map<ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList*,
                     host_memory::Tag::originals>
        originals;
    ledger::Ledger states;
    uint64_t ownership_deferrals = 0, unwrapped_lists = 0, identity_failures = 0;
    std::string output_state_basis, identity_refusal, ownership_refusal, output_state_refusal;
    bool output_state_from_committed_observation = false;
    std::string stage = "idle";
    uint64_t stage_frame = 0;
    // Measurement mode. Correctness modes leave all of this untouched.
    bool measure = false, claim_enabled = true, measurement_complete = false;
    uint64_t warmup_ms = 0, measure_ms = 0;
    uint64_t first_present_tick = 0, window_start_tick = 0, window_end_tick = 0;
    uint64_t window_presents = 0, window_claimed_at_start = 0, window_completed_at_start = 0;
    uint64_t presents = 0, last_summary_tick = 0;
    std::unique_ptr<timing::Timestamps> timestamps;
    std::unique_ptr<phoenix::GpuTiming> live_timing;
    std::vector<std::unique_ptr<phoenix::GpuTiming>> retired_timing;
    std::shared_ptr<phoenix::TimingArchive> timing_archive =
        std::make_shared<phoenix::TimingArchive>();
    phoenix::WorkerTiming worker_timing;
    phoenix::WorkerBudgetFeedback worker_budget;
    unsigned pending_timing_slot = UINT_MAX;
    // Lifecycle and footprint observation. All of it is read-only.
    ComPtr<IDXGIAdapter3> adapter;
    uint64_t swapchains_created = 0, swapchains_destroyed = 0;
    uint64_t swapchain_resizes = 0, fullscreen_transitions = 0;
    uint64_t last_backbuffer_width = 0, last_backbuffer_height = 0;
    unsigned last_backbuffer_format = 0, last_color_space = 0;
    bool last_fullscreen = false;
    uint64_t working_set_first = 0, working_set_last = 0, working_set_peak = 0;
    uint64_t private_first = 0, private_last = 0, private_peak = 0;
    uint64_t video_first = 0, video_last = 0, video_peak = 0, video_budget = 0;
    uint64_t memory_samples = 0, last_memory_tick = 0, last_series_tick = 0, series_interval = 1000;
    Json lifecycle_events = Json::array();
    Json memory_series = Json::array();
    phoenix_native::memory::Ledger owned_allocations;
    // Re-acquisition after the engine stops using the feature we claimed.
    uint64_t claim_bypassed = 0, claim_releases_by_abandonment = 0;
    unsigned abandonment_threshold = 120;
    Json device_identity = Json::object();
    std::unique_ptr<phoenix_bundle::Bundle> bundle;
    std::vector<phoenix_kernel::Binary> candidates;
    phoenix_model::WeightSource weight_source;
    phoenix_model::WeightSource configured_weight_source;
    std::string active_model_source = "configured", active_model_folder, active_model_dll, active_model_id;
    std::string active_nvidia_dll_folder;
    std::filesystem::path model_dll_path, portable_model_dll_path;
    phoenix::ModelDllMonitor model_dll_monitor;
    phoenix_model_tools::Task model_picker, model_validation;
    phoenix_embedded::Installation embedded;
    uint64_t settings_saved_at = 0;
    std::filesystem::path sandbox, output_directory, gate, bundle_path;
    reshade::api::effect_runtime* screenshot_runtime = nullptr;
    bool require_gpu_ready_file = false;
    bool configured = false, automatic = false, terminal = false, sdr_proxy = false,
         hdr_bridge = false;
    bool hdr_control_override = false;
    // Production submits each natural group as one chain with full barriers.
    std::string in_place;
    std::string native_dll_sha256, allowed_native_dll, native_dll_variant_approval;
    bool native_dll_is_reference = false;
    std::vector<uint8_t> group_batched;
    unsigned frame_limit = 32;
    unsigned capture_limit = 32;
    uint64_t generations = 0, releases = 0, resets = 0, last_sampling = 0, candidate_launches = 0;
    uint64_t claimed = 0, recorded = 0, rejected = 0, submitted = 0, unclaimed_native = 0;
    uint64_t blocked_submissions = 0;
    uint64_t budget_refusals = 0;
    uint64_t observed_creations = 0;
    uint64_t observed_evaluates = 0;
    // Observe-mode state survey: per-Evaluate visibility of the inputs and
    // output in this list's epoch. Counts only; the claim machinery reads
    // the same lookups, so a blind survey predicts a blind claim.
    uint64_t observe_state_total = 0;
    uint64_t observe_color_known = 0, observe_motion_known = 0;
    uint64_t observe_output_known = 0, observe_output_discarded = 0;
    bool observe_color_seen = false, observe_motion_seen = false;
    bool observe_output_seen = false, observe_output_discarded_seen = false;
    unsigned observe_color_state = 0, observe_motion_state = 0;
    unsigned observe_output_state = 0, observe_output_discarded_state = 0;
    uint64_t completed_retired = 0;
    std::string error;
    std::unique_ptr<phoenix_pipeline::RecoveryJournal> runtime_recovery;
    std::string recovery_notice;
    uint64_t recovery_completed = 0;
    Json frames = Json::array();
#if CNR_PHOENIX_DIAGNOSTIC
    std::vector<std::shared_ptr<diagnostic::Pair>> pairs;
    std::shared_ptr<diagnostic::Pair> measurement_output, active_pair;
    std::filesystem::path capture_directory;
    uint64_t diagnostic_native_attempted = 0, diagnostic_native_succeeded = 0;
    bool comparison_failed = false;
#endif
};
// Retained at process exit: GPU modules/resources must never be released under
// loader lock. CnrPHOENIXUnregister releases a retired lease during normal execution.
State& state() {
    static auto* result = new State;
    return *result;
}
void record_recovery_failure(const char* text) noexcept;
bool recover_initialization_failure(const char* text) noexcept;
void error(const char* text) {
    record_recovery_failure(text);
    // First error wins in the summary: every later error is collateral fired
    // by abandon paths after the cause (invalidate, then the submit block
    // reporting the invalidation it just saw). Every error still reaches
    // ReShade.log; only the summary keeps the root cause. All writers set
    // terminal alongside, so nothing clears-and-continues past a kept error.
    if (state().error.empty())
        state().error = text;
    reshade::log::message(reshade::log::level::error, text);
}
void retire() {
    auto& addon_state = state();
    if (addon_state.endpoint.session) {
        addon_state.endpoint.session->invalidate();
        addon_state.retired.push_back(addon_state.endpoint.session);
        addon_state.endpoint = {};
    }
}
void collect() {
    auto& addon_state = state();
    // Submission is observed on the engine's submitter thread; the commit it
    // authorises belongs to the recording thread and happens here or at the
    // next recording, never inside the submission callback.
    auto settle = [](const std::shared_ptr<Session>& session) {
        if (!session || session->settled())
            return;
        try {
            session->settle();
        } catch (...) { /* already invalidated */
        }
    };
    settle(addon_state.endpoint.session);
    for (const auto& session : addon_state.retired)
        settle(session);
    addon_state.route_outputs.erase(
        std::remove_if(addon_state.route_outputs.begin(), addon_state.route_outputs.end(),
                       [&](const auto& lease) {
                           return lease.session != addon_state.endpoint.session &&
                                  lease.session->can_destroy();
                       }),
        addon_state.route_outputs.end());
    addon_state.comparisons.erase(
        std::remove_if(addon_state.comparisons.begin(), addon_state.comparisons.end(),
                       [&](const auto& lease) {
                           return lease.session != addon_state.endpoint.session &&
                                  lease.session->can_destroy();
                       }),
        addon_state.comparisons.end());
    addon_state.retired.erase(
        std::remove_if(addon_state.retired.begin(), addon_state.retired.end(),
                       [&](const auto& session) {
                           if (!session->can_destroy())
                               return false;
                           for (const auto& entry : addon_state.submitted_lists)
                               if (entry.second == session)
                                   return false;
                           addon_state.candidate_launches += session->candidate_launches();
                           addon_state.completed_retired += session->completed();
                           return true;
                       }),
        addon_state.retired.end());
}
void prepare_automatic(ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*,
                       const NVSDK_NGX_Parameter*);
void capture_present_guides(ID3D12GraphicsCommandList*, const NVSDK_NGX_Parameter*);
void direct_game_frame(ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*,
                       const NVSDK_NGX_Parameter*);
void arm_input_capture(unsigned frames);
void write_capture_frame(phoenix::direct::CaptureFrame& cap);
bool gpu_gate() noexcept;
void sample_memory();
void record_event(const char* kind, Json detail);
// Records how far a claimed frame reached and flushes it before the next step,
// so evidence survives even if the step never returns.
void progress(const char* stage);
ID3D12GraphicsCommandList* resolve_native_list(ID3D12GraphicsCommandList* supplied);
bool ownership_established(ID3D12GraphicsCommandList* cmd) noexcept;
void resolve_engine_output(ID3D12GraphicsCommandList* cmd, ID3D12CommandQueue* queue,
                           const NVSDK_NGX_Parameter* params, FrameEvidence& frame);
void copy_band(ID3D12GraphicsCommandList* cmd, ID3D12Resource* source,
               D3D12_RESOURCE_STATES source_state, ID3D12Resource* output,
               D3D12_RESOURCE_STATES output_state, unsigned first, unsigned last) {
    const auto source_description = source->GetDesc(), output_description = output->GetDesc();
    check(
        source != output && source_description.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
            source_description.Format == output_description.Format &&
            source_description.Width == output_description.Width &&
            source_description.Height == output_description.Height &&
            source_description.DepthOrArraySize == 1 && output_description.DepthOrArraySize == 1 &&
            source_description.MipLevels == 1 && output_description.MipLevels == 1 &&
            source_description.SampleDesc.Count == 1 && output_description.SampleDesc.Count == 1 &&
            first < last && last <= source_description.Width,
        "Phoenix DLSS5 comparison requires matching single-subresource image formats and extents");
    phoenix_native::transition(cmd, source, source_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    phoenix_native::transition(cmd, output, output_state, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    from.pResource = source;
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    to.pResource = output;
    to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    const D3D12_BOX box{first, 0, 0, last, source_description.Height, 1};
    cmd->CopyTextureRegion(&to, first, 0, 0, &from, &box);
    phoenix_native::transition(cmd, output, D3D12_RESOURCE_STATE_COPY_DEST, output_state);
    phoenix_native::transition(cmd, source, D3D12_RESOURCE_STATE_COPY_SOURCE, source_state);
}

void poll_route_pairs() {
    auto& addon_state = state();
    for (const auto& pair : addon_state.route_pairs) {
        if (pair->poll(addon_state.route_capture, false) &&
            pair->result().at("comparison") != "byte_exact") {
            addon_state.terminal = true;
            error("Phoenix DLSS5 switch diagnostic differs from the untouched DLL");
        }
    }
}
#if CNR_PHOENIX_DIAGNOSTIC
void poll_diagnostic() {
    auto& addon_state = state();
    for (const auto& pair : addon_state.pairs) {
        const auto index = pair->result().at("pair").get<uint64_t>();
        if (pair->poll(addon_state.capture_directory, index <= addon_state.capture_limit) &&
            (pair->result().at("comparison") != "byte_exact" &&
             pair->result().at("comparison") != "bounded_error" &&
             pair->result().at("comparison") != "native_unavailable")) {
            addon_state.comparison_failed = true;
            addon_state.terminal = true;
            error(
                "phoenix DIAGNOSTIC comparison outside the error ceiling; no candidate-only promotion");
        }
    }
}
#endif

// Process and adapter footprint. Sampled, never acted upon: this add-on does
// not free, trim or resize anything in response to what it reads.
void sample_memory() {
    auto& addon_state = state();
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                             sizeof(counters))) {
        const uint64_t working = counters.WorkingSetSize, privately = counters.PrivateUsage;
        if (!addon_state.memory_samples) {
            addon_state.working_set_first = working;
            addon_state.private_first = privately;
        }
        addon_state.working_set_last = working;
        addon_state.private_last = privately;
        if (working > addon_state.working_set_peak)
            addon_state.working_set_peak = working;
        if (privately > addon_state.private_peak)
            addon_state.private_peak = privately;
    }
    if (addon_state.adapter) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (SUCCEEDED(addon_state.adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                                                                &info))) {
            if (!addon_state.memory_samples)
                addon_state.video_first = info.CurrentUsage;
            addon_state.video_last = info.CurrentUsage;
            addon_state.video_budget = info.Budget;
            if (info.CurrentUsage > addon_state.video_peak)
                addon_state.video_peak = info.CurrentUsage;
        }
    }
    ++addon_state.memory_samples;
}
// The series used to stop dead at 512 entries, so a two-hour soak recorded
// only its first few minutes and the trend that mattered was invisible. It now
// decimates instead: when it fills, every second sample is dropped and the
// interval doubles, so the series always spans the whole run within a fixed
// budget. `series_interval_ms` says what the spacing ended up being.
void sample_series(uint64_t now) {
    auto& addon_state = state();
    if (addon_state.memory_series.size() &&
        now - addon_state.last_series_tick < addon_state.series_interval)
        return;
    addon_state.last_series_tick = now;
    Json host_live = Json::object();
    {
        auto& observer_state = observation::observer();
        std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
        for (size_t iteration_index = 0; iteration_index < size_t(host_memory::Tag::count);
             ++iteration_index)
            host_live[host_memory::names[iteration_index]] =
                host_memory::counters()[iteration_index].live.load();
        host_live["observer_references"] = observation::reference_snapshot();
    }
    Json session_refs = Json::array();
    std::set<const Session*> unique_sessions;
    const auto add_session = [&](const std::shared_ptr<Session>& session) {
        if (session && unique_sessions.insert(session.get()).second)
            session_refs.push_back(session->host_references());
    };
    add_session(addon_state.endpoint.session);
    for (const auto& session : addon_state.retired)
        add_session(session);
    for (const auto& entry : addon_state.submitted_lists)
        add_session(entry.second);
    host_live["session_references"] = std::move(session_refs);
    addon_state.memory_series.push_back(Json{{"tick", now},
                                             {"working_set_bytes", addon_state.working_set_last},
                                             {"private_bytes", addon_state.private_last},
                                             {"adapter_local_bytes", addon_state.video_last},
                                             {"host_live", std::move(host_live)},
                                             {"generations", addon_state.generations},
                                             {"submitted_frames", addon_state.submitted}});
    if (addon_state.memory_series.size() < 512)
        return;
    Json kept = Json::array();
    for (size_t iteration_index = 0; iteration_index < addon_state.memory_series.size();
         iteration_index += 2)
        kept.push_back(addon_state.memory_series[iteration_index]);
    addon_state.memory_series = std::move(kept);
    addon_state.series_interval *= 2;
}
void record_event(const char* kind, Json detail) {
    auto& addon_state = state();
    if (addon_state.panel.log_level > 0)
        reshade::log::message(reshade::log::level::info,
                              (std::string("Phoenix DLSS5 ") + kind +
                               (addon_state.panel.log_level == 2 ? " " + detail.dump() : ""))
                                  .c_str());
    if (addon_state.lifecycle_events.size() >= 256)
        return;
    sample_memory();
    detail["event"] = kind;
    detail["tick"] = GetTickCount64();
    detail["generations"] = addon_state.generations;
    detail["feature_releases"] = addon_state.releases;
    detail["claimed_frames"] = addon_state.claimed;
    detail["submitted_frames"] = addon_state.submitted;
    detail["working_set_bytes"] = addon_state.working_set_last;
    detail["private_bytes"] = addon_state.private_last;
    detail["adapter_local_bytes"] = addon_state.video_last;
    addon_state.lifecycle_events.push_back(std::move(detail));
}

void progress(const char* stage) {
    auto& addon_state = state();
    addon_state.stage = stage;
    if (addon_state.measure || addon_state.interactive)
        return; // no per-step disk writes in live use
#if CNR_PHOENIX_DIAGNOSTIC
    if (addon_state.automatic)
        CnrPHOENIXSummary(
            (addon_state.output_directory / "production_game_phoenix_diagnostic_summary.json")
                .c_str());
#else
    if (addon_state.automatic)
        CnrPHOENIXSummary(
            (addon_state.output_directory / "production_game_phoenix_summary.json").c_str());
#endif
}

// Observe mode only: per-Evaluate visibility of the NR inputs and output in
// this list's epoch -- the same lookups the claim machinery reads. Counts
// and first-seen states only, all best-effort inside try/catch: the survey
// must never break forwarding.
void survey_observe_states(ID3D12GraphicsCommandList* cmd, const NVSDK_NGX_Parameter* params) {
    auto& addon_state = state();
    try {
        if (!cmd || !params)
            return;
        ++addon_state.observe_state_total;
        D3D12_RESOURCE_STATES known{};
        auto* color = parameter_resource(params, "DLSSNR.Color");
        auto* motion = parameter_resource(params, "DLSSNR.MVec");
        auto* output = parameter_resource(params, "DLSSNR.Output");
        if (color && observation::epoch_state(cmd, color, known)) {
            ++addon_state.observe_color_known;
            if (!addon_state.observe_color_seen) {
                addon_state.observe_color_seen = true;
                addon_state.observe_color_state = unsigned(known);
            }
        }
        if (motion && observation::epoch_state(cmd, motion, known)) {
            ++addon_state.observe_motion_known;
            if (!addon_state.observe_motion_seen) {
                addon_state.observe_motion_seen = true;
                addon_state.observe_motion_state = unsigned(known);
            }
        }
        if (output) {
            if (observation::epoch_state(cmd, output, known)) {
                ++addon_state.observe_output_known;
                if (!addon_state.observe_output_seen) {
                    addon_state.observe_output_seen = true;
                    addon_state.observe_output_state = unsigned(known);
                }
            } else if (observation::discarded_epoch_state(cmd, output, known)) {
                ++addon_state.observe_output_discarded;
                if (!addon_state.observe_output_discarded_seen) {
                    addon_state.observe_output_discarded_seen = true;
                    addon_state.observe_output_discarded_state = unsigned(known);
                }
            }
        }
    } catch (...) { /* survey never breaks forwarding */
    }
}
// Observe mode only: record native input/output extents and formats on the
// forward path. Raw DXGI_FORMAT ints (dxgiformat.h maps them); every key is
// best-effort, a failed Get leaves its slot 0 (UNKNOWN/absent). No GPU
// readback, no claiming, no session.
void log_native_formats(const NVSDK_NGX_Parameter* params) {
    auto& addon_state = state();
    unsigned width = 0, height = 0, color = 0, mvec = 0, output = 0, depth = 0;
    if (params) {
        const auto format_of = [&](const char* key, unsigned& out, bool extent = false) {
            void* value = nullptr;
            if (!NVSDK_NGX_SUCCEED(params->Get(key, &value)) || !value)
                return;
            const auto desc = static_cast<ID3D12Resource*>(value)->GetDesc();
            out = unsigned(desc.Format);
            if (extent) {
                width = unsigned(desc.Width);
                height = desc.Height;
            }
        };
        format_of("DLSSNR.Color", color, true);
        format_of("DLSSNR.MVec", mvec);
        format_of("DLSSNR.Output", output);
        format_of("DLSSNR.Depth", depth);
    }
    char message[256];
    std::snprintf(
        message, sizeof(message),
        "[Phoenix DLSS5 observe] native %ux%u color=%u mvec=%u output=%u depth=%u evals=%llu",
        width, height, color, mvec, output, depth, addon_state.observed_evaluates);
    reshade::log::message(reshade::log::level::info, message);
    try {
        record_event(
            "observed_native_formats",
            {{"width", width},
             {"height", height},
             {"color_format", color},
             {"mvec_format", mvec},
             {"output_format", output},
             {"depth_format", depth},
             {"evaluates", addon_state.observed_evaluates},
             {"state_surveyed", addon_state.observe_state_total},
             {"state_color_known", addon_state.observe_color_known},
             {"state_motion_known", addon_state.observe_motion_known},
             {"state_output_known", addon_state.observe_output_known},
             {"state_output_discarded", addon_state.observe_output_discarded},
             {"state_first_color", addon_state.observe_color_state},
             {"state_first_motion", addon_state.observe_motion_state},
             {"state_first_output", addon_state.observe_output_state},
             {"state_first_output_discarded", addon_state.observe_output_discarded_state}});
    } catch (...) {
    }
}
NVSDK_NGX_Result evaluate(void* instance, ID3D12GraphicsCommandList* cmd,
                          const NVSDK_NGX_Handle* feature, const NVSDK_NGX_Parameter* params,
                          PFN_NVSDK_NGX_ProgressCallback progress_callback) {
    auto& addon_state = state();
    std::unique_lock<std::recursive_mutex> lock(addon_state.mutex);
    if (!feature)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    // Native 0x18670 loads the QWORD key, not the handle object's address.
    const auto key = native_feature_key(feature);
    const bool fresh = addon_state.fresh_features.erase(key) != 0;
    // The list handed to the boundary may be a runtime proxy. State
    // observation, recording and submission matching must all name the
    // original object, while the untouched original Evaluate keeps exactly
    // the pointer the engine passed.
    ID3D12GraphicsCommandList* const supplied = cmd;
    ID3D12GraphicsCommandList* owned = nullptr;
    std::string identity_error;
    bool timing_open = false;
    if (addon_state.automatic && !addon_state.terminal) {
        try {
            owned = resolve_native_list(supplied);
        } catch (const std::exception& feature_entry) {
            identity_error = feature_entry.what();
            ++addon_state.identity_failures;
            try {
                addon_state.identity_refusal = identity_error;
            } catch (...) {
            }
        }
    }
    if (addon_state.automatic && !addon_state.terminal && !addon_state.claimed_feature &&
        addon_state.claim_enabled && gpu_gate()) {
        int reset = 0;
        if (params && !NVSDK_NGX_SUCCEED(params->Get("DLSSNR.Reset", &reset)))
            reset = 0;
        // A takeover may begin only where the candidate history can start
        // equal to the engine's, and only once this exact list's submitting
        // queue is established from real observed submissions. Otherwise the
        // frame is left entirely to the original implementation.
        if (fresh || reset != 0) {
            if (owned && ownership_established(owned)) {
                addon_state.claimed_feature = key;
                addon_state.fresh_claim = fresh;
                addon_state.claim_bypassed = 0;
            } else
                ++addon_state.ownership_deferrals;
        }
    }
    if (!addon_state.claimed_feature || *addon_state.claimed_feature != key) {
        ++addon_state.unclaimed_native;
        // Observe mode records native formats on the forward path. Past the
        // record cap the call still forwards silently below â€” never refused.
        if (addon_state.observe && addon_state.observed_evaluates < addon_state.frame_limit) {
            ++addon_state.observed_evaluates;
            survey_observe_states(owned ? owned : cmd, params);
            if (addon_state.observed_evaluates == 1 || (addon_state.observed_evaluates % 3600) == 0)
                log_native_formats(params);
        }
        // A geometry change leaves the engine evaluating a different feature
        // while ours is never released. Holding the claim forever would mean
        // one resize permanently ends the takeover, so the claim is given up
        // once the engine has clearly moved on AND everything we recorded has
        // retired: no pending recording, no outstanding list, fences complete.
        if (addon_state.claimed_feature &&
            ++addon_state.claim_bypassed >= addon_state.abandonment_threshold) {
            auto& session = addon_state.endpoint.session;
            const bool quiet = !session || (session->settled() && !session->command() &&
                                            !session->pending() && session->can_destroy());
            if (quiet) {
                retire();
                collect();
                addon_state.claimed_feature.reset();
                addon_state.claim_bypassed = 0;
                ++addon_state.claim_releases_by_abandonment;
                try {
                    record_event("claim_released_after_engine_moved_on", Json::object());
                } catch (...) {
                }
            }
        }
        const auto original = addon_state.original_evaluate;
        lock.unlock();
        return original(instance, cmd, feature, params, progress_callback);
    }
    ++addon_state.claimed;
    addon_state.stage_frame = addon_state.claimed;
    if (addon_state.automatic)
        progress("claimed");
    if (addon_state.automatic && owned)
        cmd = owned;
    if (addon_state.automatic && !addon_state.terminal &&
        (addon_state.submitted >= addon_state.frame_limit ||
         (addon_state.routing_test_period &&
          addon_state.routed_frames >= addon_state.frame_limit))) {
        ++addon_state.budget_refusals;
        return NVSDK_NGX_Result_FAIL_NotInitialized;
    }
    try {
        check(!addon_state.terminal, "phoenix bounded candidate session terminated");
        check(!addon_state.automatic || owned, identity_error.empty()
                                                   ? "phoenix Evaluate list identity unresolved"
                                                   : identity_error.c_str());
        const auto dispatch_before = addon_state.dispatch_counts;
        dispatch_audit::Scope audit_scope(addon_state.dispatch_counts);
        if (addon_state.automatic) {
            check(gpu_gate(), "Requested test GPU readiness file is absent");
            prepare_automatic(cmd, feature, params);
            progress("session_ready");
        }
        const auto endpoint = addon_state.endpoint;
        check(bool(endpoint.session), "phoenix candidate endpoint unavailable");
        check(!endpoint.session->failed(), "phoenix registered session invalidated");
        FrameEvidence evidence;
#if CNR_PHOENIX_DIAGNOSTIC
        // The per-frame contract resolve is strict everywhere except here:
        // queue/binding attribution still refuses terminally, but a routine
        // output-state refusal only withholds native capture for this frame
        // (candidate-only), exactly as on the claim and capture paths. Only
        // the check() contract form degrades; anything else still fails loud.
        try {
            check(endpoint.resolve(endpoint.context, cmd, feature, params, &evidence),
                  "phoenix frame contract unresolved");
        } catch (const std::runtime_error&) {
            check(evidence.queue != nullptr, "phoenix frame queue unattributed");
            evidence.binding_lease = observation::resolve(cmd, params, evidence, false);
        }
#else
        check(endpoint.resolve(endpoint.context, cmd, feature, params, &evidence),
              "phoenix frame contract unresolved");
#endif
        if (addon_state.automatic)
            progress("frame_contract_resolved");
        harvest_appearance(cmd,params,evidence);
#if CNR_PHOENIX_DIAGNOSTIC
        if (!addon_state.measure)
            poll_diagnostic();
        check(!addon_state.terminal, "phoenix diagnostic comparison already failed");
        check(addon_state.measure || addon_state.pairs.size() < addon_state.frame_limit,
              "phoenix diagnostic pair budget exhausted");
        unsigned outstanding = 0;
        for (const auto& pair : addon_state.pairs)
            if (!pair->completed())
                ++outstanding;
        check(outstanding < 4,
              "phoenix diagnostic four-pair in-flight limit; no GPU wait or unbounded allocation");
        if (!addon_state.measure && addon_state.capture_directory.empty()) {
            addon_state.capture_directory =
                addon_state.output_directory /
                ("production_game_phoenix_diagnostic_capture_" +
                 std::to_string(GetCurrentProcessId()) + "_" + std::to_string(GetTickCount64()));
            check(std::filesystem::create_directory(addon_state.capture_directory),
                  "phoenix diagnostic capture directory must be new");
        }
        // While measuring there is no comparison and no readback: only one
        // owned output texture is reused, so the measured cost is the network,
        // not a diagnostic capture.
        if (!addon_state.measure || !addon_state.measurement_output) {
            auto created = std::make_shared<diagnostic::Pair>(
                evidence.queue, cmd, parameter_resource(params, "DLSSNR.Output", true),
                addon_state.pairs.size() + 1);
            if (addon_state.measure)
                addon_state.measurement_output = created;
            else
                addon_state.pairs.push_back(created);
            addon_state.active_pair = created;
        } else {
            addon_state.active_pair = addon_state.measurement_output;
        }
        auto pair = addon_state.active_pair;
        diagnostic::CandidateParameters candidate_params(*params, pair->candidate_output());
        auto candidate_evidence = evidence;
        // The candidate writes into a texture this add-on created, so its state
        // is known by construction rather than observed from the engine.
        candidate_evidence.output_state = D3D12_RESOURCE_STATE_COPY_DEST;
        candidate_evidence.output_state_verified = true;
        // Native first, candidate second. On a clean list the native sees
        // exactly what it sees in observe runs (engine bindings, engine
        // states, original params, original feature history). A native
        // failure is recorded on the pair, never terminal and never
        // session-invalidating: the candidate side is independent evidence
        // (its output is known-by-construction), and abandoning the session
        // over a native cold-start would destroy it. Where the native
        // succeeds full pairs flow as before; where it refuses, pairs come
        // back candidate-only until the output-state contract admits native
        // captures.
        if (addon_state.timestamps)
            timing_open = addon_state.timestamps->begin_frame(cmd);
        ++addon_state.diagnostic_native_attempted;
        if (addon_state.automatic)
            progress("native_evaluate_started");
        if (timing_open)
            addon_state.timestamps->original_begin(cmd);
        const auto native_result =
            addon_state.original_evaluate(instance, supplied, feature, params, progress_callback);
        if (timing_open)
            addon_state.timestamps->original_end(cmd);
        if (addon_state.automatic)
            progress("native_evaluate_returned");
        const bool native_ok = bool(NVSDK_NGX_SUCCEED(native_result));
        if (native_ok) {
            ++addon_state.diagnostic_native_succeeded;
        } else {
            const auto game_controls = read_controls(
                *params, [](NVSDK_NGX_Result tag_row) { return NVSDK_NGX_SUCCEED(tag_row); });
            pair->mark_native_unavailable(static_cast<unsigned long long>(native_result),
                                          {{"enabled", game_controls.enabled},
                                           {"intensity", game_controls.intensity},
                                           {"style", game_controls.style},
                                           {"ui_correction", game_controls.ui_correction}},
                                          "native-evaluate-failed");
            if (addon_state.automatic)
                progress("native_unavailable");
        }
        if (addon_state.automatic)
            progress("candidate_recording");
        endpoint.session->record(cmd, &candidate_params, candidate_evidence);
        if (timing_open)
            addon_state.timestamps->candidate_end(cmd);
        if (addon_state.automatic)
            progress("candidate_recorded");
        if (!addon_state.measure)
            pair->capture_candidate(cmd);
        if (addon_state.automatic)
            progress("candidate_captured");
        check(!addon_state.terminal && endpoint.session->pending() && !endpoint.session->failed(),
              "phoenix diagnostic paired recording failed");
        check(addon_state.measure ||
                  parameter_resource(params, "DLSSNR.Output", true) == pair->native_output(),
              "phoenix diagnostic native output parameter changed during Evaluate");
        if (native_ok) {
            FrameEvidence native_after;
            native_after.queue = evidence.queue;
            native_after.binding_lease = observation::resolve(cmd, params, native_after, false);
            // The output-state resolve is pure observation (no GPU work), so
            // its contract refusal degrades to candidate-only instead of
            // terminal: only std::runtime_error (the check() contract form)
            // is caught -- anything else still fails loud. Where the output
            // IS established full pairs flow exactly as before.
            bool output_known = false;
            try {
                resolve_engine_output(cmd, evidence.queue, params, native_after);
                output_known = native_after.output_state_verified;
            } catch (const std::runtime_error&) {
                output_known = false;
            }
            if (!output_known) {
                pair->mark_native_unavailable(static_cast<unsigned long long>(native_result),
                                              endpoint.session->frame_controls(),
                                              "output-state-unestablished");
                if (addon_state.automatic)
                    progress("native_unavailable");
            }
            if (output_known && !addon_state.measure)
                pair->capture_native(cmd, native_after.output_state,
                                     endpoint.session->frame_controls());
        }
        ++addon_state.recorded;
        if (addon_state.automatic)
            progress("native_captured");
#else
        if (addon_state.automatic)
            progress("candidate_recording");
        check(addon_state.panel.valid(), "Phoenix DLSS5 invalid panel settings");
        endpoint.session->settle();
        const int mode =
            addon_state.routing_test_period
                ? int((addon_state.routed_frames / addon_state.routing_test_period + 3) % 4)
                : addon_state.panel.mode;
        addon_state.depth_budget.configure(
            mode == phoenix::ours && addon_state.panel.experimental &&
                addon_state.panel.automatic_budget,
            unsigned(addon_state.panel.transformer_blocks),
            unsigned(addon_state.panel.minimum_blocks), addon_state.panel.budget_target_ms(),
            addon_state.panel.budget_mode == 1, addon_state.panel.block_repeats);
        const unsigned blocks = mode == phoenix::ours && addon_state.panel.experimental &&
                                        addon_state.panel.automatic_budget
                                    ? addon_state.depth_budget.depth()
                                    : addon_state.panel.blocks_for(mode);
        const bool changed_mode = mode != addon_state.last_mode;
        if (blocks != addon_state.last_transformer_blocks)
            addon_state.live.reset_requested = true;
        if (changed_mode)
            addon_state.reset_hold = addon_state.panel.reset_after_switch;
        if (addon_state.panel.force_reset || addon_state.reset_hold > 0 ||
            (addon_state.panel.reset_every > 0 &&
             addon_state.routed_frames % unsigned(addon_state.panel.reset_every) == 0))
            addon_state.live.reset_requested = true;
        const bool reset_requested = addon_state.live.reset_requested;
        // Keep the native object's private fields; edit only keys whose exact
        // typed values can be restored. Overrides are scoped to this Evaluate.
        phoenix::ParameterEdits edits(
            *const_cast<NVSDK_NGX_Parameter*>(params),
            [](NVSDK_NGX_Result result) { return NVSDK_NGX_SUCCEED(result); });
        if (reset_requested && mode != phoenix::input)
            edits.add("DLSSNR.Reset", 1);
        if (phoenix::uses_native(mode) && addon_state.panel.uplift) {
            if (addon_state.live.enabled) {
                edits.add("DLSSNR.LocalToneStrength", addon_state.live.tone);
                edits.add("DLSSNR.LocalStructureStrength", addon_state.live.structure);
            }
            if (addon_state.live.advanced) {
                edits.add("DLSSNR.SkinStructureStrength", addon_state.live.skin);
                edits.add("DLSSNR.UseAutoMask", addon_state.live.auto_mask ? 1 : 0);
                float host_mx = 1.0f, host_my = 1.0f;
                NVSDK_NGX_SUCCEED(params->Get("DLSSNR.MVecScaleX", &host_mx));
                NVSDK_NGX_SUCCEED(params->Get("DLSSNR.MVecScaleY", &host_my));
                edits.add("DLSSNR.MVecScaleX", host_mx * addon_state.live.motion_x);
                edits.add("DLSSNR.MVecScaleY", host_my * addon_state.live.motion_y);
                if (addon_state.live.depth >= 0)
                    edits.add("DLSSNR.DepthInverted", addon_state.live.depth);
            }
            if (mode == phoenix::nvidia && addon_state.panel.native_options) {
                edits.add("DLSSNR.Intensity", addon_state.panel.intensity);
                edits.add("DLSSNR.Style", unsigned(addon_state.panel.style));
                edits.add("DLSSNR.UICorrection", addon_state.panel.ui_correction);
            }
        }
        edits.apply();
        auto* output = parameter_resource(params, "DLSSNR.Output", true);
        auto* color = parameter_resource(params, "DLSSNR.Color", true);
        const unsigned width = unsigned(output->GetDesc().Width);
        if (addon_state.live_timing && addon_state.timing_settings_dirty) {
            Json timing_settings = Json::object();
            phoenix::visit_settings(addon_state.panel, [&](const char* setting, auto& value) {
                timing_settings[setting] = value;
            });
            // Backend stays out of the key so mode rows accumulate for
            // comparison; Divider is cosmetic. Experimental and
            // TransformerBlocks change DLSS5 Phoenix cost directly, so they
            // reset the averages like any other bench-relevant setting.
            for (const char* setting : {"Backend", "Divider", "ScreenshotRequest", "CaptureBurst",
                                        "ExternalReload", "LogLevel"})
                timing_settings.erase(setting);
            addon_state.live_timing->settings(timing_settings.dump());
            addon_state.timing_settings_dirty = false;
        }
        if (addon_state.live_timing)
            addon_state.live_timing->begin(cmd, unsigned(mode), blocks, width,
                                           output->GetDesc().Height);
        ID3D12Resource* scratch = nullptr;
        std::shared_ptr<diagnostic::Pair> test_pair;
        if (mode == phoenix::split) {
            if (addon_state.routing_test_period) {
                poll_route_pairs();
                check(!addon_state.terminal, "Phoenix DLSS5 switch comparison already failed");
                check(addon_state.route_pairs.size() < 128,
                      "Phoenix DLSS5 switch comparison budget");
                if (addon_state.route_capture.empty()) {
                    addon_state.route_capture =
                        addon_state.output_directory / "phoenix_switch_capture";
                    check(std::filesystem::create_directory(addon_state.route_capture),
                          "Phoenix DLSS5 capture directory must be new");
                }
                test_pair = std::make_shared<diagnostic::Pair>(evidence.queue, cmd, output,
                                                               addon_state.route_pairs.size() + 1);
                addon_state.route_pairs.push_back(test_pair);
                scratch = test_pair->candidate_output();
            } else {
                addon_state.route_outputs.erase(
                    std::remove_if(addon_state.route_outputs.begin(),
                                   addon_state.route_outputs.end(),
                                   [&](const auto& lease) {
                                       return lease.session != endpoint.session &&
                                              lease.session->can_destroy();
                                   }),
                    addon_state.route_outputs.end());
                for (const auto& lease : addon_state.route_outputs)
                    if (lease.session == endpoint.session)
                        scratch = lease.output.Get();
                if (!scratch) {
                    check(addon_state.route_outputs.size() < 8,
                          "Phoenix DLSS5 comparison output retirement bound");
                    const auto owner = resolve_allocation_device(evidence.queue, output);
                    check(owner.resolved, "Phoenix DLSS5 comparison allocation device unresolved");
                    auto resource = phoenix_native::image(
                        owner.device.Get(), width, output->GetDesc().Height,
                        output->GetDesc().Format, D3D12_RESOURCE_STATE_COPY_DEST, false);
                    scratch = resource.Get();
                    addon_state.route_outputs.push_back({endpoint.session, std::move(resource)});
                }
            }
        }
        if (phoenix::uses_ours(mode)) {
            if (addon_state.live_timing)
                addon_state.live_timing->own_begin();
            if (addon_state.timestamps)
                timing_open = addon_state.timestamps->begin_frame(cmd);
            if (scratch) {
                diagnostic::CandidateParameters candidate_params(*params, scratch);
                auto candidate_evidence = evidence;
                candidate_evidence.output_state = D3D12_RESOURCE_STATE_COPY_DEST;
                candidate_evidence.output_state_verified = true;
                endpoint.session->record(cmd, &candidate_params, candidate_evidence,
                                         &addon_state.live, blocks);
                if (test_pair)
                    test_pair->capture_candidate(cmd);
            } else
                endpoint.session->record(cmd, params, evidence, &addon_state.live, blocks);
            ++addon_state.recorded;
            ++addon_state.depth_frames[blocks];
            if (addon_state.live_timing)
                addon_state.live_timing->own_end();
            if (timing_open)
                addon_state.timestamps->candidate_end(cmd);
        }
        if (phoenix::uses_native(mode)) {
            if (addon_state.live_timing)
                addon_state.live_timing->native_begin();
            check(NVSDK_NGX_SUCCEED(addon_state.original_evaluate(instance, supplied, feature,
                                                                  params, progress_callback)),
                  "Phoenix DLSS5 selected native Evaluate failed");
            if (addon_state.live_timing)
                addon_state.live_timing->native_end();
            if (mode == phoenix::split) {
                ++addon_state.comparison_frames[size_t(addon_state.panel.comparison)];
                FrameEvidence after;
                after.queue = evidence.queue;
                after.binding_lease = observation::resolve(cmd, params, after, false);
                resolve_engine_output(cmd, evidence.queue, params, after);
                check(after.output_state_verified && after.states_verified,
                      "Phoenix DLSS5 split output/input state unresolved after native");
                if (test_pair)
                    test_pair->capture_native(cmd, after.output_state,
                                              endpoint.session->frame_controls());
                if (addon_state.panel.comparison == 0) {
                    const unsigned divider = unsigned(float(width) * addon_state.panel.divider);
                    copy_band(cmd, color, after.color_state, output, after.output_state, 0,
                              divider / 2);
                    copy_band(cmd, scratch, D3D12_RESOURCE_STATE_COPY_DEST, output,
                              after.output_state, divider, width);
                } else {
                    phoenix::Comparison* compositor = nullptr;
                    for (const auto& lease : addon_state.comparisons)
                        if (lease.session == endpoint.session)
                            compositor = lease.compositor.get();
                    if (!compositor) {
                        check(addon_state.comparisons.size() < 8,
                              "Phoenix DLSS5 compositor retirement bound");
                        const auto owner = resolve_allocation_device(evidence.queue, output);
                        check(owner.resolved, "Phoenix DLSS5 compositor device unresolved");
                        auto created = std::make_unique<phoenix::Comparison>(
                            owner.device.Get(), width, output->GetDesc().Height,
                            output->GetDesc().Format);
                        compositor = created.get();
                        addon_state.comparisons.push_back({endpoint.session, std::move(created)});
                    }
                    compositor->record(
                        cmd, {color, output, scratch},
                        {after.color_state, after.output_state, D3D12_RESOURCE_STATE_COPY_DEST},
                        output, after.output_state, unsigned(addon_state.panel.comparison),
                        unsigned(addon_state.panel.source_a), unsigned(addon_state.panel.source_b),
                        addon_state.panel.divider, addon_state.panel.difference_gain,addon_state.panel.pipeline);
                }
            }
        } else if (mode == phoenix::input) {
            copy_band(cmd, color, evidence.color_state, output, evidence.output_state, 0, width);
        }
        edits.restore();
        if (addon_state.live_timing)
            addon_state.live_timing->end();
        if (addon_state.live.reset_requested && mode != phoenix::input)
            ++addon_state.live_resets_recorded;
        addon_state.live.recorded();
        if (addon_state.reset_hold > 0)
            --addon_state.reset_hold;
        const auto routed_own =
            addon_state.dispatch_counts.own_accepted - dispatch_before.own_accepted;
        const auto routed_native =
            addon_state.dispatch_counts.other_accepted - dispatch_before.other_accepted;
        check(routed_own ==
                  (phoenix::uses_ours(mode) ? endpoint.session->expected_candidate_launches() : 0),
              "Phoenix DLSS5 selected mode did not execute the expected own network");
        check(phoenix::uses_native(mode) ? routed_native > 0 : routed_native == 0,
              "Phoenix DLSS5 native dispatch audit disagrees with selected mode");
        ++addon_state.mode_frames[size_t(mode)];
        addon_state.mode_own[size_t(mode)] += routed_own;
        addon_state.mode_native[size_t(mode)] += routed_native;
        if (changed_mode) {
            if (addon_state.last_mode >= 0)
                ++addon_state.switches;
            if (addon_state.route_events.size() < 256)
                addon_state.route_events.push_back({{"frame", addon_state.routed_frames},
                                                    {"mode", mode},
                                                    {"reset", reset_requested},
                                                    {"own", routed_own},
                                                    {"native", routed_native}});
        }
        if (blocks != addon_state.last_transformer_blocks)
            record_event("phoenix_depth", {{"frame", addon_state.routed_frames},
                                           {"blocks", blocks},
                                           {"reset", reset_requested},
                                           {"own_launches", routed_own},
                                           {"automatic", addon_state.panel.automatic_budget}});
        addon_state.last_mode = mode;
        addon_state.last_transformer_blocks = blocks;
        ++addon_state.routed_frames;
        addon_state.displayed = addon_state.panel;
        addon_state.displayed.mode = mode;
        addon_state.display_tick = GetTickCount64();
        if (addon_state.panel.log_level == 2 && addon_state.routed_frames % 120 == 0)
            reshade::log::message(
                reshade::log::level::info,
                ("Phoenix DLSS5 frame " + std::to_string(addon_state.routed_frames) +
                 " own=" + std::to_string(routed_own) + " native=" + std::to_string(routed_native))
                    .c_str());
        if (addon_state.automatic)
            progress("candidate_recorded");
        if (!phoenix::uses_ours(mode))
            return NVSDK_NGX_Result_Success;
#endif
        if (timing_open)
            addon_state.pending_timing_slot = addon_state.timestamps->resolve(cmd);
        const auto own_dispatches =
            addon_state.dispatch_counts.own_attempted - dispatch_before.own_attempted;
        const auto other_dispatches =
            addon_state.dispatch_counts.other_attempted - dispatch_before.other_attempted;
        const unsigned expected_launches = endpoint.session->expected_candidate_launches();
        check(own_dispatches == expected_launches &&
                  addon_state.dispatch_counts.own_accepted - dispatch_before.own_accepted ==
                      expected_launches,
              "phoenix runtime CUDA observation did not confirm all expected own launches");
#if CNR_PHOENIX_DIAGNOSTIC
        check(
            other_dispatches > 0,
            "phoenix diagnostic native Evaluate produced no observed non-candidate CUDA launches");
#else
        check(addon_state.last_mode == phoenix::split ? other_dispatches > 0
                                                      : other_dispatches == 0,
              "Phoenix DLSS5 selected candidate route has unexpected native CUDA work");
#endif
        addon_state.last_sampling = endpoint.session->last_sampling();
        auto details = endpoint.session->frame_controls();
        details["record"] = addon_state.recorded;
        details["command_list"] = uint64_t(cmd);
        details["queue"] = uint64_t(evidence.queue);
        details["submitted"] = false;
        details["observed_own_cuda_launches"] = own_dispatches;
        details["observed_non_candidate_cuda_launches"] = other_dispatches;
        // A degraded frame never establishes the engine output, so the report
        // must not print the zero-initialized state as fact: null names the
        // unknown. Strict builds always verify, so their reports are unchanged.
        if (evidence.output_state_verified)
            details["restored_states"] = {unsigned(evidence.color_state),
                                          unsigned(evidence.motion_state),
                                          unsigned(evidence.output_state)};
        else
            details["restored_states"] = {unsigned(evidence.color_state),
                                          unsigned(evidence.motion_state), nullptr};
        if (addon_state.frames.size() < 32)
            addon_state.frames.push_back(std::move(details));
        if (addon_state.automatic)
            progress("frame_complete");
        return NVSDK_NGX_Result_Success;
    } catch (const std::exception& feature_entry) {
        ++addon_state.rejected;
        if (addon_state.endpoint.session)
            addon_state.endpoint.session->invalidate();
        addon_state.terminal = true;
        error(feature_entry.what());
    } catch (...) {
        ++addon_state.rejected;
        if (addon_state.endpoint.session)
            addon_state.endpoint.session->invalidate();
        addon_state.terminal = true;
        error("phoenix unknown candidate error");
    }
    if (addon_state.automatic) {
        try {
            progress("rejected");
        } catch (...) {
        }
    }
    return NVSDK_NGX_Result_FAIL_InvalidParameter;
}

static void phoenix_crumb(const char* text);
void STDMETHODCALLTYPE execute(ID3D12CommandQueue* queue, UINT count,
                               ID3D12CommandList* const* lists) {
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    try {
        if (addon_state.direct_runner)
            addon_state.direct_runner->validate_submit(queue, count, lists);
    } catch (const std::exception& feature_entry) {
        addon_state.terminal = true;
        ++addon_state.blocked_submissions;
        error(feature_entry.what());
        return;
    }
    // Observed submission ownership, recorded for every batch in the process
    // from the first registered queue onward, so a claimed frame can only
    // start once its own list already has a proven submitting queue.
    for (UINT iteration_index = 0; iteration_index < count; ++iteration_index) {
        auto* list = static_cast<ID3D12GraphicsCommandList*>(lists[iteration_index]);
        if (!list)
            continue;
        auto& owner = addon_state.ownership[list];
        if (owner.submissions && owner.queue != queue)
            owner.ambiguous = true;
        owner.queue = queue;
        ++owner.submissions;
    }
    const auto session = addon_state.endpoint.session;
    auto* cmd = session ? session->command() : nullptr;
    // Snapshot BEFORE any uncertain_submission() below invalidates the
    // transaction: the post-block read would always show the invalidation,
    // never the cause.
    const bool entry_txn_ok = session && cmd && session->submission_matches(queue, cmd);
    unsigned matches = 0;
    bool blocked = false, duplicate = false;
    for (UINT iteration_index = 0; cmd && iteration_index < count; ++iteration_index)
        if (lists[iteration_index] == cmd) {
            ++matches;
            if (session)
                session->note_submit_sighting();
        }
    for (UINT iteration_index = 0; iteration_index < count; ++iteration_index) {
        auto* list = static_cast<ID3D12GraphicsCommandList*>(lists[iteration_index]);
        auto found = addon_state.submitted_lists.find(list);
        if (found != addon_state.submitted_lists.end()) {
            duplicate = true;
            found->second->uncertain_submission();
            blocked = true;
        }
        for (const auto& previous : addon_state.retired)
            if (previous->command() == list) {
                previous->uncertain_submission();
                blocked = true;
            }
    }
    if (matches && (matches != 1 || !session->submission_matches(queue, cmd))) {
        session->uncertain_submission();
        blocked = true;
    }
    if (blocked) {
        ++addon_state.blocked_submissions;
        addon_state.terminal = true;
        std::string detail =
            "phoenix blocked invalid/duplicate/wrong-thread/wrong-queue candidate submission; entire batch not executed";
        if (session && cmd) {
            detail += " matches=" + std::to_string(matches);
            detail += " batch=" + std::to_string(count);
            detail += " submit_queue_is_session_queue=" +
                      std::string(queue == session->queue() ? "1" : "0");
            detail += " transaction_matches=" +
                      std::string(session->submission_matches(queue, cmd) ? "1" : "0");
            detail += " session_queue=" + std::to_string(uint64_t(session->queue()));
            detail += " exec_queue=" + std::to_string(uint64_t(queue));
            detail += " session_command=" + std::to_string(uint64_t(session->command()));
            detail += " txn_command=" + std::to_string(session->pending_transaction_command());
            detail += " txn_queue=" + std::to_string(session->pending_transaction_queue());
            detail += " entry_txn_ok=" + std::string(entry_txn_ok ? "1" : "0");
            detail += " duplicate=" + std::string(duplicate ? "1" : "0");
            detail += " sightings=" + std::to_string(session->submit_sightings());
        } else
            detail += " no_session_command";
        error(detail.c_str());
        return;
    }
    if (matches) {
        try {
            addon_state.submitted_lists.emplace(cmd, session);
        } catch (...) {
            session->uncertain_submission();
            addon_state.terminal = true;
            ++addon_state.blocked_submissions;
            error("phoenix unable to retain submission lease; entire batch not executed");
            return;
        }
    }
    try {
        if (addon_state.live_timing)
            addon_state.live_timing->validate_submission(queue, count, lists);
        for (const auto& timer : addon_state.retired_timing)
            timer->validate_submission(queue, count, lists);
    } catch (const std::exception& feature_entry) {
        if (session)
            session->uncertain_submission();
        addon_state.terminal = true;
        ++addon_state.blocked_submissions;
        error(feature_entry.what());
        return;
    }
    // Submit-time breadcrumb: when the driver dies inside original_execute,
    // the pre line without its post line names the exact batch.
    const bool direct_batch =
        addon_state.direct_runner && addon_state.direct_runner->pending_in_batch(count, lists);
    if (direct_batch && addon_state.panel.log_level == 2) {
        reshade::log::message(
            reshade::log::level::info,
            ("[Phoenix DLSS5] execute: submitting claimed game list batch=" + std::to_string(count))
                .c_str());
        char crumb[96];
        snprintf(crumb, sizeof(crumb), "execute: submitting claimed game list batch=%u", count);
        phoenix_crumb(crumb);
    }
    addon_state.original_execute(queue, count, lists);
    try { if (addon_state.present_guides) addon_state.present_guides->submitted(queue,count,lists); }
    catch(const std::exception& e) {addon_state.terminal=true;error(e.what());return;}
    if (direct_batch && addon_state.panel.log_level == 2) {
        reshade::log::message(reshade::log::level::info,
                              "[Phoenix DLSS5] execute: claimed game list submitted");
        phoenix_crumb("execute: claimed game list submitted");
    }
    try {
        if (addon_state.direct_runner &&
            addon_state.direct_runner->after_submit(queue, count, lists))
            ++addon_state.submitted;
    } catch (const std::exception& feature_entry) {
        addon_state.terminal = true;
        error(feature_entry.what());
        return;
    }
    try {
        if (addon_state.live_timing)
            addon_state.live_timing->submitted(queue, count, lists);
        for (const auto& timer : addon_state.retired_timing)
            timer->submitted(queue, count, lists);
    } catch (const std::exception& feature_entry) {
        if (session)
            session->uncertain_submission();
        addon_state.terminal = true;
        error(feature_entry.what());
        return;
    }
    for (UINT iteration_index = 0; iteration_index < count; ++iteration_index)
        if (lists[iteration_index])
            addon_state.states.submitted(uint64_t(lists[iteration_index]), uint64_t(queue));
    if (!matches)
        return;
    try {
        check(matches == 1, "phoenix candidate command submitted more than once in a batch");
        session->after_submit(queue, cmd);
        ++addon_state.submitted;
        for (const auto& pair : addon_state.route_pairs)
            if (!pair->submitted() && pair->command() == cmd)
                pair->after_submit(queue, cmd);
        if (addon_state.timestamps && addon_state.pending_timing_slot != UINT_MAX) {
            addon_state.timestamps->submitted(addon_state.pending_timing_slot, session->signaled());
            addon_state.pending_timing_slot = UINT_MAX;
        }
#if CNR_PHOENIX_DIAGNOSTIC
        if (!addon_state.measure) {
            bool pair_found = false;
            for (const auto& pair : addon_state.pairs)
                if (!pair->submitted() && pair->command() == cmd) {
                    check(!pair_found, "phoenix multiple diagnostic pairs in one pending list");
                    pair->after_submit(queue, cmd);
                    pair_found = true;
                }
            check(pair_found, "phoenix submitted candidate lacks diagnostic pair");
        }
#endif
        for (auto entry_iterator = addon_state.frames.rbegin();
             entry_iterator != addon_state.frames.rend(); ++entry_iterator) {
            if (entry_iterator->at("command_list") == uint64_t(cmd) &&
                entry_iterator->at("submitted") == false) {
                (*entry_iterator)["submitted"] = true;
                break;
            }
        }
    } catch (const std::exception& feature_entry) {
        session->uncertain_submission();
        addon_state.terminal = true;
        error(feature_entry.what());
    } catch (...) {
        session->uncertain_submission();
        addon_state.terminal = true;
        error("phoenix submission callback failed");
    }
}

NVSDK_NGX_Result release(const NVSDK_NGX_Handle* feature) {
    auto& addon_state = state();
    {
        std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
        if (feature)
            addon_state.fresh_features.erase(native_feature_key(feature));
        if (feature && addon_state.claimed_feature &&
            *addon_state.claimed_feature == native_feature_key(feature)) {
            ++addon_state.releases;
            retire();
            addon_state.claimed_feature.reset();
            addon_state.claim_bypassed = 0;
            try {
                record_event("claimed_feature_released", Json::object());
            } catch (...) {
            }
        }
    }
    // Original feature creation/destruction remains engine-owned. This is not
    // an inference fallback and is reported as a remaining native dependency.
    return addon_state.original_release(feature);
}
NVSDK_NGX_Result create(void* instance, ID3D12GraphicsCommandList* cmd,
                        const NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** handle) {
    auto& addon_state = state();
    const auto result = addon_state.original_create(instance, cmd, params, handle);
    if (NVSDK_NGX_SUCCEED(result) && handle && *handle) {
        std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
        try {
            check(addon_state.fresh_features.size() < 1024,
                  "phoenix unobserved feature creation limit");
            addon_state.fresh_features.insert(native_feature_key(*handle));
            ++addon_state.observed_creations;
        } catch (const std::exception& feature_entry) {
            addon_state.terminal = true;
            error(feature_entry.what());
        }
    }
    return result;
}

bool hook(void* target, void* detour, void** original) {
    auto status = MH_Initialize();
    if (status != MH_OK && status != MH_ERROR_ALREADY_INITIALIZED)
        return false;
    if (MH_CreateHook(target, detour, original) != MH_OK)
        return false;
    if (MH_EnableHook(target) == MH_OK)
        return true;
    MH_RemoveHook(target);
    *original = nullptr;
    return false;
}

// NGX export interposition (renodx replacement, stage A). One template
// instantiation per NGX copy slot, so each detour forwards into its own
// copy's original. Strictly observers: forward first, record sizes/handles
// in a catch-all, never claim, never refuse, never alter parameters. The
// internal-offset hooks keep doing all processing; a call passes through
// at most one observer (this layer) and one processor (that layer).
void ngx_creator_name(void* caller, char* name, size_t size) {
    if (!name || !size)
        return;
    name[0] = 0;
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(caller), &module) ||
        !module)
        return;
    wchar_t path[1024]{};
    GetModuleFileNameW(module, path, 1024);
    const wchar_t* base = wcsrchr(path, L'\\');
    base = base ? base + 1 : path;
    WideCharToMultiByte(CP_UTF8, 0, base, -1, name, int(size), nullptr, nullptr);
}
template <size_t Slot>
NVSDK_NGX_Result NVSDK_CONV ngx_export_create(ID3D12GraphicsCommandList* cmd,
                                              NVSDK_NGX_Feature feature_id,
                                              NVSDK_NGX_Parameter* params,
                                              NVSDK_NGX_Handle** handle) {
    auto& addon_state = state();
    auto next =
        Slot < addon_state.ngx_copies.size() ? addon_state.ngx_copies[Slot].create : nullptr;
    if (!next)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    const auto result = next(cmd, feature_id, params, handle);
    try {
        // Evaluates arrive on engine task threads; the table is shared, so
        // every touch serializes on the addon lock (recursive: a detour may
        // run on a thread already inside). Unlocked map access here wedged
        // Bonsai past its deadline with zero completed frames.
        std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
        if (NVSDK_NGX_SUCCEED(result) && handle && *handle) {
            unsigned in_w = 0, in_h = 0, out_w = 0, out_h = 0;
            if (params) {
                if (!NVSDK_NGX_SUCCEED(params->Get(NVSDK_NGX_Parameter_Width, &in_w)))
                    in_w = 0;
                if (!NVSDK_NGX_SUCCEED(params->Get(NVSDK_NGX_Parameter_Height, &in_h)))
                    in_h = 0;
                if (!NVSDK_NGX_SUCCEED(params->Get(NVSDK_NGX_Parameter_OutWidth, &out_w)))
                    out_w = 0;
                if (!NVSDK_NGX_SUCCEED(params->Get(NVSDK_NGX_Parameter_OutHeight, &out_h)))
                    out_h = 0;
            }
            char creator[128]{};
            ngx_creator_name(_ReturnAddress(), creator, sizeof creator);
            const auto key = native_feature_key(*handle);
            const bool first = addon_state.ngx_observed.find(key) == nullptr;
            unsigned create_flags = 0;
            if (params)
                params->Get("DLSS.Feature.Create.Flags", &create_flags);
            addon_state.ngx_observed.note_create(key, unsigned(feature_id), in_w, in_h, out_w,
                                                 out_h, creator, create_flags);
            ++addon_state.ngx_features_seen;
            if (first)
                reshade::log::message(reshade::log::level::info,
                                      ("[Phoenix DLSS5] NGX feature created: id=" +
                                       std::to_string(unsigned(feature_id)) + " " +
                                       std::to_string(in_w) + "x" + std::to_string(in_h) + " -> " +
                                       std::to_string(out_w) + "x" + std::to_string(out_h) +
                                       " by " + creator)
                                          .c_str());
            // Renodx-replacement recipe harvest: the first create per RR/NR
            // id dumps every documented create-time key, so the self-drive
            // stage mirrors the working foreign recipe instead of guessing
            // it. Bounded: one line per id per process, observers only.
            if (first && params && (unsigned(feature_id) == 13 || unsigned(feature_id) == 18)) {
                float float_value = 0;
                auto getu = [&](const char* name, unsigned& out) {
                    out = 0xFFFFFFFFu;
                    unsigned parameter_value = 0;
                    if (NVSDK_NGX_SUCCEED(params->Get(name, &parameter_value)))
                        out = parameter_value;
                };
                auto geti = [&](const char* name, int& out) {
                    out = -2147483647 - 1;
                    int parameter_value = 0;
                    if (NVSDK_NGX_SUCCEED(params->Get(name, &parameter_value)))
                        out = parameter_value;
                };
                unsigned input_width = 0, input_height = 0, output_width = 0, output_height = 0,
                         creation_node_mask = 0, visibility_node_mask = 0;
                int quality = 0, flags = 0, subrects = 0, rtx = 0;
                float sharp = 0;
                getu(NVSDK_NGX_Parameter_Width, input_width);
                getu(NVSDK_NGX_Parameter_Height, input_height);
                getu(NVSDK_NGX_Parameter_OutWidth, output_width);
                getu(NVSDK_NGX_Parameter_OutHeight, output_height);
                getu(NVSDK_NGX_Parameter_CreationNodeMask, creation_node_mask);
                getu(NVSDK_NGX_Parameter_VisibilityNodeMask, visibility_node_mask);
                geti(NVSDK_NGX_Parameter_PerfQualityValue, quality);
                geti(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, flags);
                geti(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, subrects);
                geti(NVSDK_NGX_Parameter_RTXValue, rtx);
                if (NVSDK_NGX_SUCCEED(params->Get(NVSDK_NGX_Parameter_Sharpness, &float_value)))
                    sharp = float_value;
                char message[512];
                const uint64_t mod =
                    (Slot < addon_state.ngx_copies.size() && addon_state.ngx_copies[Slot].module)
                        ? uint64_t(addon_state.ngx_copies[Slot].module)
                        : 0;
                ComPtr<ID3D12Device> create_device;
                if (cmd)
                    cmd->GetDevice(IID_PPV_ARGS(&create_device));
                std::snprintf(
                    message, sizeof(message),
                    "[Phoenix DLSS5] NGX create recipe id=%u slot=%llu mod=0x%llx wh=%u,%u out=%u,%u nodes=%u,%u quality=%d flags=0x%x subrects=%d rtx=%d sharp=%f cmd=0x%llx dev=0x%llx",
                    unsigned(feature_id), uint64_t(Slot), mod, input_width, input_height,
                    output_width, output_height, creation_node_mask, visibility_node_mask, quality,
                    unsigned(flags), subrects, rtx, double(sharp), uint64_t(cmd),
                    uint64_t(create_device.Get()));
                reshade::log::message(reshade::log::level::info, message);
            }
        }
    } catch (...) {
    }
    return result;
}
template <size_t Slot>
NVSDK_NGX_Result NVSDK_CONV ngx_export_evaluate(ID3D12GraphicsCommandList* cmd,
                                                const NVSDK_NGX_Handle* feature,
                                                const NVSDK_NGX_Parameter* params,
                                                PFN_NVSDK_NGX_ProgressCallback progress_callback) {
    auto& addon_state = state();
    auto next =
        Slot < addon_state.ngx_copies.size() ? addon_state.ngx_copies[Slot].evaluate : nullptr;
    if (!next)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    // Several NGX DLLs forward through each other. Only the outermost return
    // sees the final SR/RR output; never process it twice.
    struct Depth {
        Depth() {
            ++phoenix_ngx_depth;
        }
        ~Depth() {
            --phoenix_ngx_depth;
        }
    } depth;
    try {
        std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
        const auto* obs = feature ? addon_state.ngx_observed.find(native_feature_key(feature)) : nullptr;
        if (phoenix_ngx_depth == 1 && addon_state.direct_game && !addon_state.direct_feeder &&
            obs && ngx_shim::is_direct_game_feature(obs->feature_id) &&
            addon_state.panel.processing.enabled && addon_state.panel.processing.hook == 1) {
            struct RenderScope { RenderScope(){phoenix_render_stage=true;} ~RenderScope(){phoenix_render_stage=false;} } scope;
            direct_game_frame(cmd, feature, params);
        }
    } catch (...) {}
    const auto result = next(cmd, feature, params, progress_callback);
    try {
        std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
        if (feature)
            addon_state.ngx_observed.note_evaluate(native_feature_key(feature));
    } catch (...) {
    }
    // Self-drive piggyback: after a game RR evaluate was forwarded, run our
    // own NR evaluate on the same list (renodx's inline pattern). Serialized
    // on the addon lock like all frame work; the dlssnr export we call is
    // unhooked, so our own call never re-enters this detour.
    try {
        if (feature) {
            std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
            const auto* obs = addon_state.ngx_observed.find(native_feature_key(feature));
            if (addon_state.direct_game && !addon_state.direct_feeder && phoenix_ngx_depth == 1 && NVSDK_NGX_SUCCEED(result) &&
                obs && ngx_shim::is_direct_game_feature(obs->feature_id)) {
                if (addon_state.panel.processing.enabled && addon_state.panel.processing.hook == 3)
                    capture_present_guides(cmd, params);
                else if (!addon_state.panel.processing.enabled || addon_state.panel.processing.hook == 2)
                    direct_game_frame(cmd, feature, params);
                if (addon_state.panel.mode == phoenix::nvidia)
                    phoenix_crumb("nr-frame: evaluate returned to game");
            }
        }
    } catch (...) {
    }
    return result;
}
template <size_t Slot> NVSDK_NGX_Result NVSDK_CONV ngx_export_release(NVSDK_NGX_Handle* feature) {
    auto& addon_state = state();
    auto next =
        Slot < addon_state.ngx_copies.size() ? addon_state.ngx_copies[Slot].release : nullptr;
    if (!next)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    const auto result = next(feature);
    try {
        std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
        if (feature)
            addon_state.ngx_observed.note_release(native_feature_key(feature));
    } catch (...) {
    }
    return result;
}
template <size_t Slot>
bool hook_ngx_copy(State::NgxCopy& slot, void* create_addr, void* evaluate_addr,
                   void* release_addr) {
    if (!hook(create_addr, reinterpret_cast<void*>(ngx_export_create<Slot>),
              reinterpret_cast<void**>(&slot.create)))
        return false;
    if (!hook(evaluate_addr, reinterpret_cast<void*>(ngx_export_evaluate<Slot>),
              reinterpret_cast<void**>(&slot.evaluate))) {
        MH_RemoveHook(create_addr);
        slot.create = nullptr;
        return false;
    }
    if (!hook(release_addr, reinterpret_cast<void*>(ngx_export_release<Slot>),
              reinterpret_cast<void**>(&slot.release))) {
        MH_RemoveHook(create_addr);
        MH_RemoveHook(evaluate_addr);
        slot.create = nullptr;
        slot.evaluate = nullptr;
        return false;
    }
    return true;
}
// Best-effort and order-free: copies load at different times (driver copy
// at boot, game copies with Streamline, NR runtime on first RR use), so
// every install path retries; hooked slots are skipped. Failures here are
// never terminal -- observation degrades to the internal hooks alone.
void install_ngx_exports() {
    auto& addon_state = state();
    // Surface partition (hard-won). renodx recognizes its own inline calls
    // by caller identity; any OUTER evaluate detour on the copy serving its
    // private features (nvngx_dlssnr.dll) makes its own calls look foreign,
    // its inline driving stops after one evaluation, and no claimable frame
    // ever arrives (Bonsai 012-015/017/018: zero frames, deadline; 016 with
    // hooks off: green). So dlssnr exports are NEVER hooked here -- that
    // copy is renodx's private surface. These three copies carry the game's
    // own creates/evaluates, where we install INNER (addon-load first,
    // event retries after): an inner observer never blinds an outer
    // dispatcher, and Detours re-arms restore our jump untouched.
    // Our dlssnr visibility stays exactly what it always was: the
    // internal-offset hooks, a surface renodx never touches.
    static const wchar_t* names[] = {L"_nvngx.dll", L"nvngx_dlss.dll", L"nvngx_dlssd.dll"};
    for (size_t iteration_index = 0; iteration_index < 3; ++iteration_index) {
        State::NgxCopy& slot = addon_state.ngx_copies[iteration_index];
        if (slot.hooked)
            continue;
        HMODULE module = GetModuleHandleW(names[iteration_index]);
        if (!module)
            continue;
        void* create_addr = GetProcAddress(module, "NVSDK_NGX_D3D12_CreateFeature");
        void* evaluate_addr = GetProcAddress(module, "NVSDK_NGX_D3D12_EvaluateFeature");
        if (!evaluate_addr)
            evaluate_addr = GetProcAddress(module, "NVSDK_NGX_D3D12_EvaluateFeature_C");
        void* release_addr = GetProcAddress(module, "NVSDK_NGX_D3D12_ReleaseFeature");
        if (!create_addr || !evaluate_addr || !release_addr)
            continue;
        bool success = false;
        switch (iteration_index) {
        case 0:
            success = hook_ngx_copy<0>(slot, create_addr, evaluate_addr, release_addr);
            break;
        case 1:
            success = hook_ngx_copy<1>(slot, create_addr, evaluate_addr, release_addr);
            break;
        default:
            success = hook_ngx_copy<2>(slot, create_addr, evaluate_addr, release_addr);
            break;
        }
        if (!success) {
            slot = State::NgxCopy{};
            continue;
        }
        slot.module = module;
        slot.hooked = true;
        ++addon_state.ngx_export_hooks;
        std::string narrow;
        for (const wchar_t* module_name_character = names[iteration_index]; *module_name_character;
             ++module_name_character)
            narrow.push_back(char(*module_name_character));
        reshade::log::message(reshade::log::level::info,
                              ("[Phoenix DLSS5] NGX export interposition on " + narrow).c_str());
    }
}
// renodx parity: the signed NR runtime must be present before any RR/NR
// feature can exist. The game-local copy is loaded (never configured away);
// absence is not an error here -- internal hooks install if/when it loads.
// Streamline tag harvest (renodx replacement, sl stage). Sole hooker on
// sl.interposer (renodx is NGX-hooks-only), so no order games and no
// caller-identity hazard by construction. Forward first, translate tags to
// shim primitives under the addon lock, never refuse, never alter.
sl::Result sl_export_evaluate(sl::Feature feature, const sl::FrameToken& frame,
                              const sl::BaseStructure** inputs, uint32_t numInputs,
                              sl::CommandBuffer* cmdBuffer) {
    auto& addon_state = state();
    auto next = addon_state.sl_evaluate;
    if (!next)
        return sl::Result::eErrorInvalidParameter;
    const auto result = next(feature, frame, inputs, numInputs, cmdBuffer);
    try {
        if (feature == 0u || feature == sl_shim::kFeatureDLSSRR || feature == 1004u) {
            unsigned out_w = 0, out_h = 0;
            if (inputs)
                for (uint32_t iteration_index = 0;
                     iteration_index < numInputs && inputs[iteration_index]; ++iteration_index) {
                    const sl::BaseStructure* base = inputs[iteration_index];
                    if (base->structType == sl::DLSSOptions::s_structType) {
                        const auto* observed_options = static_cast<const sl::DLSSOptions*>(base);
                        out_w = observed_options->outputWidth;
                        out_h = observed_options->outputHeight;
                        break;
                    }
                    if (base->structType == sl::DLSSDOptions::s_structType) {
                        const auto* observed_options = static_cast<const sl::DLSSDOptions*>(base);
                        out_w = observed_options->outputWidth;
                        out_h = observed_options->outputHeight;
                        break;
                    }
                }
            std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
            const uint32_t idx = frame;
            const bool first = addon_state.sl_observed.find(uint64_t(feature)) == nullptr;
            addon_state.sl_observed.note_evaluate(uint64_t(feature), idx, out_w, out_h);
            ++addon_state.sl_evaluates_seen;
            if (first)
                reshade::log::message(reshade::log::level::info,
                                      ("[Phoenix DLSS5] SL feature evaluated: id=" +
                                       std::to_string(uint64_t(feature)) + " out=" +
                                       std::to_string(out_w) + "x" + std::to_string(out_h))
                                          .c_str());
        }
    } catch (...) {
    }
    return result;
}
// Translate the host tag table into owned observation records. Prefer
// immutable resource dimensions when the host supplies only native pointers;
// keep the tag extent as the fallback, then join tags to the frame/viewport.
sl::Result sl_export_settag(const sl::FrameToken& frame, const sl::ViewportHandle& viewport,
                            const sl::ResourceTag* tags, uint32_t numTags,
                            sl::CommandBuffer* cmdBuffer) {
    auto& addon_state = state();
    auto next = addon_state.sl_settag;
    if (!next)
        return sl::Result::eErrorInvalidParameter;
    std::vector<sl::ResourceTag> filtered;
    {
        std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
        const auto& p=addon_state.panel.processing;
        if(tags && numTags && numTags<256 && addon_state.direct_game && !addon_state.direct_feeder &&
            !addon_state.terminal && gpu_gate() && addon_state.panel.uplift &&
            addon_state.panel.mode!=phoenix::input && p.enabled && p.hook==3 && p.block_hudless) {
            filtered.assign(tags,tags+numTags);
            for(auto& tag:filtered)if(tag.type==sl_shim::kTagHUDLessColor)tag.resource=nullptr;
            tags=filtered.data();
        }
    }
    const auto result = next(frame, viewport, tags, numTags, cmdBuffer);
    try {
        if (tags && numTags && numTags < 256) {
            sl_shim::SlTag buf[64];
            size_t tag_count = 0;
            for (uint32_t iteration_index = 0; iteration_index < numTags && tag_count < 64;
                 ++iteration_index) {
                const sl::ResourceTag& resource_tag = tags[iteration_index];
                if (!resource_tag.resource || !resource_tag.resource->native)
                    continue;
                sl_shim::SlTag translated_tag;
                translated_tag.resource_key =
                    reinterpret_cast<uint64_t>(resource_tag.resource->native);
                translated_tag.buffer_type = resource_tag.type;
                translated_tag.w = resource_tag.resource->width;
                translated_tag.h = resource_tag.resource->height;
                translated_tag.format = resource_tag.resource->nativeFormat;
                translated_tag.ex_w = resource_tag.extent.width;
                translated_tag.ex_h = resource_tag.extent.height;
                // RE9 tags zero-filled Resource structs (pointers only): the
                // geometry lives on the resource itself. GetDesc is immutable
                // and safe here; the struct/extent stay as fallback.
                auto* native = reinterpret_cast<ID3D12Resource*>(resource_tag.resource->native);
                const auto desc = native->GetDesc();
                if (desc.Width && desc.Height) {
                    translated_tag.w = unsigned(desc.Width);
                    translated_tag.h = unsigned(desc.Height);
                    translated_tag.format = unsigned(desc.Format);
                }
                buf[tag_count++] = translated_tag;
            }
            if (tag_count) {
                std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
                addon_state.sl_observed.note_tags(uint32_t(frame), buf, tag_count,
                                                  uint32_t(viewport));
                addon_state.sl_tags_seen += uint64_t(tag_count);
            }
        }
    } catch (...) {
    }
    return result;
}
// Translate the host tag table into owned observation records. Prefer
// immutable resource dimensions when the host supplies only native pointers;
// keep the tag extent as the fallback, then join tags to the frame/viewport.
sl::Result sl_export_settag_old(const sl::ViewportHandle& viewport, const sl::ResourceTag* tags,
                                uint32_t numTags, sl::CommandBuffer* cmdBuffer) {
    auto& addon_state = state();
    auto next = addon_state.sl_settag_old;
    if (!next)
        return sl::Result::eErrorInvalidParameter;
    std::vector<sl::ResourceTag> filtered;
    {
        std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
        const auto& p=addon_state.panel.processing;
        if(tags && numTags && numTags<256 && addon_state.direct_game && !addon_state.direct_feeder &&
            !addon_state.terminal && gpu_gate() && addon_state.panel.uplift &&
            addon_state.panel.mode!=phoenix::input && p.enabled && p.hook==3 && p.block_hudless) {
            filtered.assign(tags,tags+numTags);
            for(auto& tag:filtered)if(tag.type==sl_shim::kTagHUDLessColor)tag.resource=nullptr;
            tags=filtered.data();
        }
    }
    const auto result = next(viewport, tags, numTags, cmdBuffer);
    try {
        if (tags && numTags && numTags < 256) {
            sl_shim::SlTag buf[64];
            size_t tag_count = 0;
            for (uint32_t iteration_index = 0; iteration_index < numTags && tag_count < 64;
                 ++iteration_index) {
                const sl::ResourceTag& resource_tag = tags[iteration_index];
                if (!resource_tag.resource || !resource_tag.resource->native)
                    continue;
                sl_shim::SlTag translated_tag;
                translated_tag.resource_key =
                    reinterpret_cast<uint64_t>(resource_tag.resource->native);
                translated_tag.buffer_type = resource_tag.type;
                translated_tag.w = resource_tag.resource->width;
                translated_tag.h = resource_tag.resource->height;
                translated_tag.format = resource_tag.resource->nativeFormat;
                translated_tag.ex_w = resource_tag.extent.width;
                translated_tag.ex_h = resource_tag.extent.height;
                auto* native = reinterpret_cast<ID3D12Resource*>(resource_tag.resource->native);
                const auto desc = native->GetDesc();
                if (desc.Width && desc.Height) {
                    translated_tag.w = unsigned(desc.Width);
                    translated_tag.h = unsigned(desc.Height);
                    translated_tag.format = unsigned(desc.Format);
                }
                buf[tag_count++] = translated_tag;
            }
            if (tag_count) {
                std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
                addon_state.sl_observed.note_tags_global(buf, tag_count);
                addon_state.sl_tags_seen += uint64_t(tag_count);
            }
        }
    } catch (...) {
    }
    return result;
}
void install_sl_hooks() {
    auto& addon_state = state();
    if (addon_state.sl_hooked)
        return;
    HMODULE module = GetModuleHandleW(L"sl.interposer.dll");
    if (!module)
        return;
    void* eval_addr = GetProcAddress(module, "slEvaluateFeature");
    void* tag_addr = GetProcAddress(module, "slSetTagForFrame");
    if (!eval_addr || !tag_addr)
        return;
    if (!hook(eval_addr, reinterpret_cast<void*>(sl_export_evaluate),
              reinterpret_cast<void**>(&addon_state.sl_evaluate)))
        return;
    if (!hook(tag_addr, reinterpret_cast<void*>(sl_export_settag),
              reinterpret_cast<void**>(&addon_state.sl_settag))) {
        MH_RemoveHook(eval_addr);
        addon_state.sl_evaluate = nullptr;
        return;
    }
    // Deprecated frame-less tagging is optional: games that tag per frame
    // never export it, and its absence must not refuse the harvest.
    void* oldtag_addr = GetProcAddress(module, "slSetTag");
    if (oldtag_addr)
        hook(oldtag_addr, reinterpret_cast<void*>(sl_export_settag_old),
             reinterpret_cast<void**>(&addon_state.sl_settag_old));
    addon_state.sl_module = module;
    addon_state.sl_hooked = true;
    ++addon_state.sl_export_hooks;
    reshade::log::message(reshade::log::level::info,
                          "[Phoenix DLSS5] Streamline tag harvest on sl.interposer.dll");
}
using NgxAllocParamsFn = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter**);
void preload_nr_runtime();
ID3D12CommandQueue* resolve_queue(ID3D12GraphicsCommandList* cmd);
// Create recipe per feature id. Sizes come from the live sl template.
// Flags: SuperSampling takes the public DLSS5-Feeder set
// (MVLowRes|AutoExposure|DepthInverted); id 13 takes the harvested renodx
// recipe (run 032: IsHDR|MVLowRes|DepthInverted). Quality is DLAA throughout
// (native-res: in == out). Subrects stay unset (renodx never sets them).
// SEH-guarded NGX calls (public DLSS5-Feeder pattern): a foreign runtime
// fault must never take the game down. Kept in dedicated functions because
// __try cannot share a frame with unwound C++ locals. A fault reports Fail
// and nulls the handle; nothing is submitted.
static NVSDK_NGX_Result guarded_ngx_create(State::NgxCreateFn create,
                                           ID3D12GraphicsCommandList* cmd,
                                           NVSDK_NGX_Feature feature_id,
                                           NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** handle) {
    if (!create || !handle)
        return NVSDK_NGX_Result_Fail;
    __try {
        return create(cmd, feature_id, params, handle);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *handle = nullptr;
        return NVSDK_NGX_Result_Fail;
    }
}
static NVSDK_NGX_Result guarded_ngx_evaluate(State::NgxEvaluateFn evaluate,
                                             ID3D12GraphicsCommandList* cmd,
                                             const NVSDK_NGX_Handle* feature,
                                             const NVSDK_NGX_Parameter* params) {
    if (!evaluate)
        return NVSDK_NGX_Result_Fail;
    __try {
        return evaluate(cmd, feature, params, nullptr);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return NVSDK_NGX_Result_Fail;
    }
}
using NrInitExtFn = NVSDK_NGX_Result(NVSDK_CONV*)(unsigned long long, const wchar_t*, ID3D12Device*,
                                                  NVSDK_NGX_Version, const NVSDK_NGX_Parameter*);
static NVSDK_NGX_Result guarded_nr_init(NrInitExtFn init, unsigned long long app,
                                        const wchar_t* path, ID3D12Device* dev,
                                        NVSDK_NGX_Parameter* params, DWORD* caught_exception = nullptr) {
    if(caught_exception)*caught_exception=0;
    if (!init)
        return NVSDK_NGX_Result_Fail;
    NVSDK_NGX_Result tag_row = NVSDK_NGX_Result_Fail;
    __try {
        tag_row = init(app, path, dev, NVSDK_NGX_Version_API, params);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        if(caught_exception)*caught_exception=GetExceptionCode();
        tag_row = NVSDK_NGX_Result_Fail;
    }
    return tag_row;
}
// NGX core Init (4-arg, no params): used only when driving RR(13) refuses to
// create on the game's own core init. Same SEH discipline as above.
using CoreReqFn = NVSDK_NGX_Result(NVSDK_CONV*)(IDXGIAdapter*,
                                                const NVSDK_NGX_FeatureDiscoveryInfo*,
                                                NVSDK_NGX_FeatureRequirement*);
static NVSDK_NGX_Result guarded_core_req(CoreReqFn request_function, IDXGIAdapter* adapter,
                                         const NVSDK_NGX_FeatureDiscoveryInfo* discovery_info,
                                         NVSDK_NGX_FeatureRequirement* req) {
    if (!request_function || !adapter || !discovery_info || !req)
        return NVSDK_NGX_Result_Fail;
    NVSDK_NGX_Result result = NVSDK_NGX_Result_Fail;
    __try {
        result = request_function(adapter, discovery_info, req);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        result = NVSDK_NGX_Result_Fail;
    }
    return result;
}
using CoreInitFn = NVSDK_NGX_Result(NVSDK_CONV*)(unsigned long long, const wchar_t*, ID3D12Device*,
                                                 NVSDK_NGX_Version);
static NVSDK_NGX_Result guarded_core_init(CoreInitFn init, unsigned long long app,
                                          const wchar_t* path, ID3D12Device* dev) {
    if (!init)
        return NVSDK_NGX_Result_Fail;
    NVSDK_NGX_Result tag_row = NVSDK_NGX_Result_Fail;
    __try {
        tag_row = init(app, path, dev, NVSDK_NGX_Version_API);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        tag_row = NVSDK_NGX_Result_Fail;
    }
    return tag_row;
}
// Create (once per geometry) through nvngx_dlssnr.dll's own exports: the
// feature runtime DOES export the CreateFeature surface (Init_Ext,
// CreateFeature, EvaluateFeature, ReleaseFeature), and must be driven
// directly, never through the NGX core. Output is RGBA16F UAV, matching the
// engine's linear domain the game color already lives in.
void selfdrive_direct_ensure(ID3D12GraphicsCommandList* cmd, ID3D12Device* device, unsigned width,
                             unsigned height, int want_bridge, int want_hdr, bool want_autoexp,
                             int want_fid, unsigned host_flags) {
    auto& addon_state = state();
    auto& selfdrive_state = addon_state.selfdrive;
    const auto preset = phoenix::nvidia_preset(addon_state.panel.nvidia_model,
        addon_state.panel.model_id.empty() ? addon_state.weight_source.model().id : addon_state.panel.model_id);
    const auto source_choice = addon_state.panel.model_source + "\n" + addon_state.panel.model_folder + "\n" + addon_state.panel.model_dll + "\n" + addon_state.panel.nvidia_dll_folder;
    const auto requested_contract=Json::array({want_fid,width,height,want_bridge,want_hdr,want_autoexp,
        preset,addon_state.panel.refinement_passes,source_choice}).dump();
    if (selfdrive_state.requested_contract != requested_contract) {
        selfdrive_state.requested_contract=requested_contract;
        selfdrive_state.failed = false;
        selfdrive_state.abandons = 0;
        selfdrive_state.attempts = 0;
        selfdrive_state.retry_after_tick = 0;
        selfdrive_state.req_queried = false;
        selfdrive_state.requested_feature = want_fid;
    }
    if (selfdrive_state.failed)
        return;
    check((!GetModuleHandleW(L"renodx-dlss5.addon64") && !GetModuleHandleW(L"renodx-dlss.addon64")),
          "phoenix NR self-host refuses a foreign NR driver");
    check(device && width && height, "phoenix NR self-host geometry missing");
    check(want_fid == 13 || want_fid == 18, "phoenix NR feature id unsupported");
    const bool want_core = (want_fid == 13);
    if (selfdrive_state.attempts == 0) {
        char emsg[192]{};
        snprintf(
            emsg, sizeof(emsg),
            "[Phoenix DLSS5] self-drive: ensure entry fid=%d want_core=%d inited=%d live_fid=%u handle=%p attempts=%u core_init=%d",
            want_fid, want_core ? 1 : 0, selfdrive_state.inited ? 1 : 0, selfdrive_state.feature_id,
            selfdrive_state.handle, selfdrive_state.attempts,
            selfdrive_state.core_init_attempted ? 1 : 0);
        reshade::log::message(reshade::log::level::warning, emsg);
    }
    wchar_t dir[MAX_PATH]{};
    GetModuleFileNameW(nullptr, dir, MAX_PATH);
    if (wchar_t* slash = wcsrchr(dir, L'\\'))
        *(slash + 1) = 0;
    // Geometry, bridge-domain, contract-flag or feature changes re-arm
    // failure into a fresh attempt: the create flags are fixed at creation.
    if (selfdrive_state.inited && selfdrive_state.handle) {
        if (selfdrive_state.in_w == width && selfdrive_state.in_h == height &&
            selfdrive_state.bridge == want_bridge && selfdrive_state.hdr_mode == want_hdr &&
            selfdrive_state.auto_exp == (want_autoexp ? 1 : 0) &&
            selfdrive_state.feature_id == unsigned(want_fid) && selfdrive_state.preset == preset &&
            selfdrive_state.passes == unsigned(addon_state.panel.refinement_passes) &&
            selfdrive_state.source_choice == source_choice)
            return;
        check(!addon_state.direct_runner || addon_state.direct_runner->idle(),
              "NVIDIA feature change waits for previous frame retirement");
        phoenix_model_tools::retire_optional_feature(selfdrive_state, true, [&](auto* handle) {
            check(selfdrive_state.release_fn != nullptr, "NVIDIA release entry unavailable");
            check(NVSDK_NGX_SUCCEED(selfdrive_state.release_fn(handle)), "NVIDIA feature retirement failed");
        });
        selfdrive_state.core_init_attempted = false;
        reshade::log::message(reshade::log::level::info,
                              "[Phoenix DLSS5] self-drive: geometry changed, recreating");
    } else if (selfdrive_state.inited && selfdrive_state.attempts == 0) {
        // A pending feature switch with no live handle (never created or a
        // past refusal): reset so the new feature gets its attempt below.
        selfdrive_state.inited = false;
        selfdrive_state.failed = false;
        selfdrive_state.attempts = 0;
        selfdrive_state.core_init_attempted = false;
    }
    // The NGX core allocates parameters for both routes and hosts RR(13).
    // Keep the selected module alive for the lifetime of its parameters.
    if (!selfdrive_state.core_module) {
        // Use the core already selected and initialized by the game. DriverStore
        // can contain several older drivers; allocating parameters from the
        // first installed copy can mix incompatible NGX instances in one process.
        GetModuleHandleExW(0, L"_nvngx.dll", &selfdrive_state.core_module);
        wchar_t core[MAX_PATH]{};
        if (!selfdrive_state.core_module) {
        WIN32_FIND_DATAW found{};
        HANDLE search = FindFirstFileW(
            L"C:\\Windows\\System32\\DriverStore\\FileRepository\\nv_dispi.inf_*", &found);
        if (search != INVALID_HANDLE_VALUE) {
            do {
                if (!(found.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
                    continue;
                swprintf_s(core,
                           L"C:\\Windows\\System32\\DriverStore\\FileRepository\\%ls\\_nvngx.dll",
                           found.cFileName);
                if (GetFileAttributesW(core) != INVALID_FILE_ATTRIBUTES)
                    break;
                core[0] = 0;
            } while (FindNextFileW(search, &found));
            FindClose(search);
        }
            check(core[0], "phoenix NGX core absent from the driver store");
            selfdrive_state.core_module = LoadLibraryW(core);
        }
        check(selfdrive_state.core_module, "phoenix NGX core failed to load");
        GetModuleFileNameW(selfdrive_state.core_module, core, MAX_PATH);
        reshade::log::message(reshade::log::level::info,
            ("[Phoenix DLSS5] self-drive NGX core: " + std::filesystem::path(core).string()).c_str());
        WIN32_FILE_ATTRIBUTE_DATA cfad{};
        if (GetFileAttributesExW(core, GetFileExInfoStandard, &cfad) &&
            !(cfad.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
            selfdrive_state.dll_bytes = (uint64_t(cfad.nFileSizeHigh) << 32) | cfad.nFileSizeLow;
        auto alloc = reinterpret_cast<NgxAllocParamsFn>(
            GetProcAddress(selfdrive_state.core_module, "NVSDK_NGX_D3D12_AllocateParameters"));
        check(alloc, "phoenix NGX core lacks AllocateParameters");
        check(NVSDK_NGX_SUCCEED(alloc(&selfdrive_state.params)) && selfdrive_state.params,
              "phoenix NR parameter allocation failed");
    }
    // Point the call surface at the selected runtime. Feature 18 drives the
    // dlssnr runtime's own NR network (lazily loaded, own Init_Ext);
    // feature 13 drives Ray Reconstruction through the NGX core, whose game
    // init normally suffices (own core Init only as a fallback at create).
    if (want_core) {
        selfdrive_state.create_fn = reinterpret_cast<State::NgxCreateFn>(
            GetProcAddress(selfdrive_state.core_module, "NVSDK_NGX_D3D12_CreateFeature"));
        selfdrive_state.evaluate_fn = reinterpret_cast<State::NgxEvaluateFn>(
            GetProcAddress(selfdrive_state.core_module, "NVSDK_NGX_D3D12_EvaluateFeature"));
        selfdrive_state.release_fn = reinterpret_cast<State::NgxReleaseFn>(
            GetProcAddress(selfdrive_state.core_module, "NVSDK_NGX_D3D12_ReleaseFeature"));
        check(selfdrive_state.create_fn && selfdrive_state.evaluate_fn &&
                  selfdrive_state.release_fn,
              "phoenix NGX core lacks feature exports");
    } else {
        const auto selected_source=phoenix_model_tools::selection(addon_state.panel, addon_state.configured_weight_source);
        auto dll=selected_source.dll.empty()
            ? phoenix_model::find_nvidia_dll({}, std::filesystem::path(dir),
                std::filesystem::u8path(addon_state.panel.nvidia_dll_folder))
            : selected_source.dll;
        check(!dll.empty(), "NVIDIA DLSS5 runtime not found locally or in the configured search folder");
        if(!selected_source.dll.empty()) {
            check(phoenix_model::known_model_container(selected_source.dll_sha256) &&
                  phoenix_embedded::file_digest(selected_source.dll)==selected_source.dll_sha256,
                  "Selected NVIDIA runtime is not an authenticated known DLL");
            dll=selected_source.dll;
        }
        if(selfdrive_state.nr_module) {
            wchar_t loaded_path[32768]{};
            check(GetModuleFileNameW(selfdrive_state.nr_module,loaded_path,32768) &&
                  std::filesystem::equivalent(std::filesystem::path(loaded_path),dll),
                  "NVIDIA runtime DLL selection changed; restart before using the newly selected library");
        }
        if (!selfdrive_state.nr_module) {
            WIN32_FILE_ATTRIBUTE_DATA fad{};
            check(GetFileAttributesExW(dll.c_str(), GetFileExInfoStandard, &fad) &&
                      !(fad.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY),
                  "NVIDIA DLSS5 runtime is missing or is not a file");
            selfdrive_state.dll_bytes = (uint64_t(fad.nFileSizeHigh) << 32) | fad.nFileSizeLow;
            selfdrive_state.nr_module = LoadLibraryW(dll.c_str());
            check(selfdrive_state.nr_module, "phoenix NR runtime failed to load");
        }
        check(preset != 2 || FindResourceW(selfdrive_state.nr_module,L"WEIGHTS_CC_SMALL_FP8",MAKEINTRESOURCEW(10)),
              "Loaded NVIDIA DLL has no Lightweight FP8 model; select the 310.9.16 DLL and restart");
        if (!selfdrive_state.forward_module) {
            const auto wrapper = addon_state.embedded.runtime / L"phoenix-nvngx.dll";
            selfdrive_state.forward_module = LoadLibraryExW(wrapper.c_str(), nullptr,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            check(selfdrive_state.forward_module, "Phoenix DLSS5 NR call wrapper is missing; rebuild the embedded addon");
            auto load = reinterpret_cast<BOOL(WINAPI*)(const wchar_t*)>(
                GetProcAddress(selfdrive_state.forward_module, "PhoenixNrLoad"));
            wchar_t model_path[32768]{};
            check(GetModuleFileNameW(selfdrive_state.nr_module, model_path, 32768) != 0 &&
                load && load(model_path), "Phoenix DLSS5 NR call wrapper cannot load the selected NVIDIA DLSS5 runtime");
        }
        // Loading a module is not initialization. A refused Init_Ext must not
        // make subsequent frames skip initialization and call CreateFeature.
        if(!selfdrive_state.nr_initialized) {
            auto init_fn = reinterpret_cast<NrInitExtFn>(
                GetProcAddress(selfdrive_state.forward_module, "PhoenixNrInit"));
            check(init_fn, "phoenix NR runtime lacks Init_Ext");
            DWORD caught_exception=0;
            const auto result=guarded_nr_init(init_fn,0x1000000ULL,dir,device,selfdrive_state.params,&caught_exception);
            if(!NVSDK_NGX_SUCCEED(result)) {
                char reason[160]{};snprintf(reason,sizeof(reason),"phoenix NR Init refused (result=0x%08X, exception=0x%08lX)",unsigned(result),caught_exception);
                throw std::runtime_error(reason);
            }
            selfdrive_state.nr_initialized=true;
            reshade::log::message(reshade::log::level::info,"[Phoenix DLSS5] self-drive: NR runtime initialized");
        }
        selfdrive_state.create_fn = reinterpret_cast<State::NgxCreateFn>(
            GetProcAddress(selfdrive_state.forward_module, "PhoenixNrCreate"));
        selfdrive_state.evaluate_fn = reinterpret_cast<State::NgxEvaluateFn>(
            GetProcAddress(selfdrive_state.forward_module, "PhoenixNrEvaluate"));
        selfdrive_state.release_fn = reinterpret_cast<State::NgxReleaseFn>(
            GetProcAddress(selfdrive_state.forward_module, "PhoenixNrRelease"));
        check(selfdrive_state.create_fn && selfdrive_state.evaluate_fn &&
                  selfdrive_state.release_fn,
              "phoenix NR runtime lacks feature exports");
    }
    if (want_core) {
        // One-time read-only verdict from the runtime itself (nested, plain
        // gates): is RR supported in our context, and if not, which bit?
        if (!selfdrive_state.req_queried) {
            selfdrive_state.req_queried = true;
            ComPtr<IDXGIAdapter> adapter;
            ComPtr<IDXGIFactory4> factory;
            if (device && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
                factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter));
            auto req_fn = reinterpret_cast<CoreReqFn>(GetProcAddress(
                selfdrive_state.core_module, "NVSDK_NGX_D3D12_GetFeatureRequirements"));
            NVSDK_NGX_FeatureDiscoveryInfo discovery_info{};
            discovery_info.SDKVersion = NVSDK_NGX_Version_API;
            discovery_info.FeatureID = static_cast<NVSDK_NGX_Feature>(want_fid);
            discovery_info.Identifier.IdentifierType =
                NVSDK_NGX_Application_Identifier_Type_Project_Id;
            discovery_info.Identifier.v.ProjectDesc.ProjectId =
                "a0f57b54-1daf-4934-90ae-c4035c19df04";
            discovery_info.Identifier.v.ProjectDesc.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
            discovery_info.Identifier.v.ProjectDesc.EngineVersion = "1.0";
            discovery_info.ApplicationDataPath = dir;
            NVSDK_NGX_FeatureRequirement req{};
            const auto query_result =
                guarded_core_req(req_fn, adapter.Get(), &discovery_info, &req);
            char qmsg[192]{};
            snprintf(
                qmsg, sizeof(qmsg),
                "[Phoenix DLSS5] self-drive: RR requirements result=0x%08llX support_flags=0x%X (0=supported on success) arch=0x%X",
                uint64_t(query_result), unsigned(req.FeatureSupported), req.MinHWArchitecture);
            reshade::log::message(reshade::log::level::warning, qmsg);
        }
        // The successful support query does not establish the missing RR
        // guide contract. direct_103 creation failed and the game stopped
        // presenting before a live switch back could run. Do not repeat
        // that mutating experiment on the game's active NGX context.
        check(
            false,
            "RR self-drive unavailable: post-DLSS inputs lack RR guides; select native NR feature 18");
    }
    selfdrive_state.device = device;
    selfdrive_state.inited = true;
    if (!selfdrive_state.output || selfdrive_state.bridge != want_bridge)
        selfdrive_state.output =
            phoenix_native::image(device, width, height, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
    // Create recipe (probe-verified at 640x384 and 1920x1080): the network's
    // own working size at 1:1 with the default preset hint. No upscaling
    // flags: the network enhances, it does not upscale.
    selfdrive_state.params->Reset();
    selfdrive_state.params->Set("CreationNodeMask", 1u);
    selfdrive_state.params->Set("VisibilityNodeMask", 1u);
    selfdrive_state.params->Set("DLSSNR.Width", width);
    selfdrive_state.params->Set("DLSSNR.Height", height);
    selfdrive_state.params->Set("DLSSNR.InputWidth", width);
    selfdrive_state.params->Set("DLSSNR.InputHeight", height);
    selfdrive_state.params->Set("DLSSNR.OutputWidth", width);
    selfdrive_state.params->Set("DLSSNR.OutputHeight", height);
    selfdrive_state.params->Set("DLSSNR.Output.Width", width);
    selfdrive_state.params->Set("DLSSNR.Output.Height", height);
    selfdrive_state.params->Set("DLSSNR.Upscaling", 0u);
    selfdrive_state.params->Set("DLSSNR.Scale", 1.0f);
    selfdrive_state.params->Set("DLSSNR.ScalingRatio", 1.0f);
    selfdrive_state.params->Set("MV.Scale.X", 1.0f);
    selfdrive_state.params->Set("MV.Scale.Y", 1.0f);
    selfdrive_state.params->Set("DLSSNR.Hint.Render.Preset", preset);
    // The game creates its RR at DLAA quality (5); MaxPerf (0) is refused for
    // 13 with InvalidParameter. 18 keeps the proven zero.
    selfdrive_state.params->Set("PerfQualityValue", want_core ? 5u : 0u);
    selfdrive_state.params->Set("DLSSNR.Hdr", (unsigned)want_hdr);
    selfdrive_state.params->Set("DLSSNR.SDR", (unsigned)(want_hdr ? 0 : 1));
    selfdrive_state.params->Set("DLSSNR.AutoExposure", (unsigned)(want_autoexp ? 1 : 0));
    selfdrive_state.params->Set("Width", width);
    selfdrive_state.params->Set("Height", height);
    selfdrive_state.params->Set("InPreExposure", 1.0f);
    selfdrive_state.params->Set("InExposureScale", 1.0f);
    selfdrive_state.params->Set("NVSDK_NGX_Parameter_PreExposure", 1.0f);
    selfdrive_state.params->Set("NVSDK_NGX_Parameter_ExposureScale", 1.0f);
    // Create flags mirror the live-harvested game RR recipe for 13
    // (IsHDR|MVLowRes|DepthInverted = 0xb at 1920x1080 DLAA); 18 keeps the
    // proven zero it always created with. Only the id selects the network.
    selfdrive_state.params->Set("DLSS.Feature.Create.Flags",
                                want_core ? 0xbu : (want_bridge ? host_flags | 0x41u : host_flags));
    if (want_core) {
        // The NGX core validates the standard output-size keys (the dlssnr
        // runtime derives them from DLSSNR.* instead); the game recipe also
        // carries Sharpness 0.35 and RTXValue 0 at DLAA.
        selfdrive_state.params->Set(NVSDK_NGX_Parameter_OutWidth, width);
        selfdrive_state.params->Set(NVSDK_NGX_Parameter_OutHeight, height);
        selfdrive_state.params->Set(NVSDK_NGX_Parameter_Sharpness, 0.35f);
        selfdrive_state.params->Set(NVSDK_NGX_Parameter_RTXValue, 0u);
        // Match NGX_D3D12_CREATE_DLSSD_EXT, including typed RR-specific keys.
        selfdrive_state.params->Set("PerfQualityValue", 5);
        selfdrive_state.params->Set("DLSS.Feature.Create.Flags", 0xb);
        selfdrive_state.params->Set("DLSS.Enable.Output.Subrects", 0);
        selfdrive_state.params->Set("DLSS.Denoise.Mode", 1);
        selfdrive_state.params->Set("DLSS.Roughness.Mode", 0u);
        selfdrive_state.params->Set("DLSS.Use.HW.Depth", 1u);
    }
    // No private core Init: the game owns the core init (its app id is the one
    // entitled to RR), and a second init under our arbitrary id poisons the
    // context for our own create. Rely on the game's init; the retry cooldown
    // covers creates that race ahead of it.
    ++selfdrive_state.attempts;
    addon_state.feeder_nr_initialization_recorded = addon_state.direct_feeder || phoenix_present_stage;
    selfdrive_state.last_create_result =
        guarded_ngx_create(selfdrive_state.create_fn, cmd, static_cast<NVSDK_NGX_Feature>(want_fid),
                           selfdrive_state.params, &selfdrive_state.handle);
    {
        char cmsg2[192]{};
        snprintf(cmsg2, sizeof(cmsg2),
                 "[Phoenix DLSS5] self-drive: RR diag fid=%d coreres=%d attempted=%d", want_fid,
                 want_core ? 1 : 0, selfdrive_state.core_init_attempted ? 1 : 0);
        reshade::log::message(reshade::log::level::warning, cmsg2);
    }
    if ((!NVSDK_NGX_SUCCEED(selfdrive_state.last_create_result) || !selfdrive_state.handle) &&
        selfdrive_state.attempts <= 3) {
        char fmsg[160]{};
        snprintf(
            fmsg, sizeof(fmsg),
            "[Phoenix DLSS5] self-drive: NR feature %d create result=0x%08llX handle=%p (attempt %u)",
            want_fid, selfdrive_state.last_create_result, selfdrive_state.handle,
            selfdrive_state.attempts);
        reshade::log::message(reshade::log::level::warning, fmsg);
    }
    check(NVSDK_NGX_SUCCEED(selfdrive_state.last_create_result) && selfdrive_state.handle,
          "phoenix NR feature creation refused");
    // Each additional evaluate owns a distinct temporal feature and output.
    // Repeatedly evaluating one feature would advance its history within a frame.
    for (unsigned index = 1; index < unsigned(addon_state.panel.refinement_passes); ++index) {
        selfdrive_state.refinement.push_back({});
        auto& pass = selfdrive_state.refinement.back();
        const auto result = guarded_ngx_create(selfdrive_state.create_fn, cmd,
            static_cast<NVSDK_NGX_Feature>(want_fid), selfdrive_state.params, &pass.handle);
        check(NVSDK_NGX_SUCCEED(result) && pass.handle, "NVIDIA refinement feature creation refused");
        pass.output = phoenix_native::image(device, width, height, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
    }
    ++selfdrive_state.creates;
    addon_state.feeder_nr_initialization_recorded = addon_state.direct_feeder || phoenix_present_stage;
    selfdrive_state.source_choice = source_choice;
    selfdrive_state.preset = preset;
    selfdrive_state.passes = unsigned(addon_state.panel.refinement_passes);
    selfdrive_state.feature_id = unsigned(want_fid);
    selfdrive_state.in_w = width;
    selfdrive_state.in_h = height;
    selfdrive_state.bridge = want_bridge;
    selfdrive_state.hdr_mode = want_hdr;
    selfdrive_state.auto_exp = want_autoexp ? 1 : 0;
    selfdrive_state.warmup = 8;
    selfdrive_state.first = true;
    {
        char cmsg[128]{};
        snprintf(cmsg, sizeof(cmsg), "[Phoenix DLSS5] self-drive: NR feature %d created %ux%u%s",
                 want_fid, width, height, want_core ? " via NGX core" : " via dlssnr runtime");
        reshade::log::message(reshade::log::level::info, cmsg);
    }
}
void selfdrive_direct_run(ID3D12GraphicsCommandList* cmd, ID3D12Resource* proxy,
                          ID3D12Resource* motion, D3D12_RESOURCE_STATES motion_state, int reset,
                          float mvsx, float mvsy, float intensity, float tone, float structure,
                          float skin, unsigned automask, unsigned style, unsigned uicorr,
                          float globaltone, float exposure, float exposure_scale, float jitter_x,
                          float jitter_y, unsigned feature_flags, ID3D12Resource* depth,
                          D3D12_RESOURCE_STATES depth_state, int inverted, unsigned out_w,
                          unsigned out_h) {
    auto& addon_state = state();
    auto& selfdrive_state = addon_state.selfdrive;
    check(selfdrive_state.inited && selfdrive_state.handle && selfdrive_state.evaluate_fn &&
              selfdrive_state.params && selfdrive_state.output,
          "phoenix NR feature not ready");
    check(
        selfdrive_state.feature_id == 18,
        "RR evaluation requires pre-denoise G-buffer guides; use native NR feature 18 for post-DLSS comparison");
    auto* parameters = selfdrive_state.params;
    parameters->Reset();
    parameters->Set("DLSSNR.Color", proxy);
    parameters->Set("Color", proxy);
    parameters->Set("DLSSNR.Output", selfdrive_state.output.Get());
    parameters->Set("Output", selfdrive_state.output.Get());
    parameters->Set("DLSSNR.MVec", motion);
    parameters->Set("MVec", motion);
    parameters->Set("MotionVectors", motion);
    parameters->Set("DLSSNR.ColorSubrectBaseX", 0u);
    parameters->Set("DLSSNR.ColorSubrectBaseY", 0u);
    parameters->Set("DLSSNR.ColorSubrectWidth", selfdrive_state.in_w);
    parameters->Set("DLSSNR.ColorSubrectHeight", selfdrive_state.in_h);
    parameters->Set("DLSSNR.MVecSubrectBaseX", 0u);
    parameters->Set("DLSSNR.MVecSubrectBaseY", 0u);
    parameters->Set("DLSSNR.MVecSubrectWidth", selfdrive_state.in_w);
    parameters->Set("DLSSNR.MVecSubrectHeight", selfdrive_state.in_h);
    parameters->Set("DLSSNR.OutputSubrectBaseX", 0u);
    parameters->Set("DLSSNR.OutputSubrectBaseY", 0u);
    parameters->Set("DLSSNR.OutputSubrectWidth", selfdrive_state.in_w);
    parameters->Set("DLSSNR.OutputSubrectHeight", selfdrive_state.in_h);
    parameters->Set("DLSSNR.MVecScaleX", mvsx);
    parameters->Set("DLSSNR.MVecScaleY", mvsy);
    parameters->Set("DLSSNR.ScalingRatio", 1.0f);
    parameters->Set("DLSSNR.Scale", 1.0f);
    parameters->Set("DLSSNR.OutputWidth", out_w);
    parameters->Set("DLSSNR.OutputHeight", out_h);
    parameters->Set("DLSSNR.Output.Width", out_w);
    parameters->Set("DLSSNR.Output.Height", out_h);
    if (inverted >= 0)
        parameters->Set("DLSSNR.DepthInverted", (unsigned)inverted);
    parameters->Set("DLSS.Feature.Create.Flags", feature_flags);
    parameters->Set("DLSSNR.Depth", depth);
    if (depth) {
        const auto depth_description = depth->GetDesc();
        parameters->Set("DLSSNR.DepthSubrectBaseX", 0u);
        parameters->Set("DLSSNR.DepthSubrectBaseY", 0u);
        parameters->Set("DLSSNR.DepthSubrectWidth", unsigned(depth_description.Width));
        parameters->Set("DLSSNR.DepthSubrectHeight", depth_description.Height);
    }
    parameters->Set("DLSSNR.Enabled", 1u);
    parameters->Set("DLSSNR.Reset", (unsigned)reset);
    parameters->Set("DLSSNR.Intensity", intensity);
    // Caller-supplied effective values (configured override or reference
    // default): the panel finally drives our own feature, not fixed profile.
    parameters->Set("DLSSNR.LocalToneStrength", tone);
    parameters->Set("DLSSNR.LocalStructureStrength", structure);
    parameters->Set("DLSSNR.SkinStructureStrength", skin);
    parameters->Set("DLSSNR.UseAutoMask", automask);
    parameters->Set("DLSSNR.Style", style);
    parameters->Set("DLSSNR.UICorrection", uicorr);
    parameters->Set("DLSSNR.GlobalToneStrength", globaltone);
    parameters->Set("DLSSNR.Hdr", unsigned(selfdrive_state.hdr_mode));
    parameters->Set("DLSSNR.SDR", unsigned(selfdrive_state.hdr_mode ? 0 : 1));
    parameters->Set("DLSSNR.Jitter.Offset.X", jitter_x);
    parameters->Set("DLSSNR.Jitter.Offset.Y", jitter_y);
    parameters->Set("Jitter.Offset.X", jitter_x);
    parameters->Set("Jitter.Offset.Y", jitter_y);
    parameters->Set("DLSS.Indicator.Invert.X.Axis", 0u);
    parameters->Set("DLSS.Indicator.Invert.Y.Axis", 0u);
    parameters->Set("Width", selfdrive_state.in_w);
    parameters->Set("Height", selfdrive_state.in_h);
    parameters->Set("DLSSNR.ControlMask", static_cast<ID3D12Resource*>(nullptr));
    parameters->Set("DLSSNR.UI", static_cast<ID3D12Resource*>(nullptr));
    parameters->Set("DLSSNR.Backbuffer", static_cast<ID3D12Resource*>(nullptr));
    parameters->Set("DLSS.Pre.Exposure", exposure);
    parameters->Set("DLSS.Exposure.Scale", exposure_scale);
    phoenix_native::transition(cmd, motion, motion_state,
                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (depth)
        phoenix_native::transition(cmd, depth, depth_state,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    selfdrive_state.last_evaluate_result =
        guarded_ngx_evaluate(selfdrive_state.evaluate_fn, cmd, selfdrive_state.handle, parameters);
    check(NVSDK_NGX_SUCCEED(selfdrive_state.last_evaluate_result), "NVIDIA first pass refused");
    auto* previous = selfdrive_state.output.Get();
    for (auto& pass : selfdrive_state.refinement) {
        phoenix_native::transition(cmd, previous, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        parameters->Set("DLSSNR.Color", previous);
        parameters->Set("Color", previous);
        parameters->Set("DLSSNR.Output", pass.output.Get());
        parameters->Set("Output", pass.output.Get());
        selfdrive_state.last_evaluate_result =
            guarded_ngx_evaluate(selfdrive_state.evaluate_fn, cmd, pass.handle, parameters);
        phoenix_native::transition(cmd, previous, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        check(NVSDK_NGX_SUCCEED(selfdrive_state.last_evaluate_result), "NVIDIA refinement pass refused");
        previous = pass.output.Get();
    }
    if (depth)
        phoenix_native::transition(cmd, depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   depth_state);
    phoenix_native::transition(cmd, motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                               motion_state);
    check(NVSDK_NGX_SUCCEED(selfdrive_state.last_evaluate_result), "phoenix NR evaluate refused");
    // The runtime leaves the output in UAV (probe-verified); the resolve
    // passes bracket it to readable around their dispatch.
    if (selfdrive_state.first) {
        selfdrive_state.first = false;
        reshade::log::message(reshade::log::level::info,
                              "[Phoenix DLSS5] self-drive: first NR evaluate submitted");
    }
    ++selfdrive_state.evaluates;
    if (selfdrive_state.evaluates == 1 || selfdrive_state.evaluates % 600 == 0) {
        char msg[288]{};
        snprintf(
            msg, sizeof(msg),
            "[Phoenix DLSS5] self-drive: NR evaluate #%llu intensity=%.2f tone=%.2f struct=%.2f skin=%.2f style=%u gtone=%.2f expo=%.3f jit=%.4f,%.4f depth=%d inv=%d reset=%d mvscale=%.3f,%.3f",
            selfdrive_state.evaluates, double(intensity), double(tone), double(structure),
            double(skin), style, double(globaltone), double(exposure), double(jitter_x),
            double(jitter_y), depth ? 1 : 0, inverted, reset, double(mvsx), double(mvsy));
        reshade::log::message(reshade::log::level::info, msg);
    }
}
// (Removed: the RR-piggyback entry never ran. The direct game path drives
// selfdrive_direct_ensure/selfdrive_direct_run explicitly per frame.)
// (Removed: preload by search-path name is superseded by the direct stage,
// which loads the runtime by absolute game-directory path.)
NvAPI_Status __cdecl observe_dispatch(ID3D12GraphicsCommandList* cmd,
                                      const NVAPI_CU_KERNEL_LAUNCH_PARAMS* launches, NvU32 count) {
    dispatch_audit::Call observed(count);
    const auto result = state().original_dispatch(cmd, launches, count);
    if (result == NVAPI_OK)
        observed.accepted();
    return result;
}
void install_dispatch_audit() {
    auto& addon_state = state();
    if (addon_state.original_dispatch)
        return;
    auto module = GetModuleHandleW(phoenix_native::nvapi_library_name());
    check(module != nullptr,
          "phoenix native feature has not loaded NVAPI for dispatch observation");
    auto query =
        reinterpret_cast<void*(__cdecl*)(unsigned)>(GetProcAddress(module, "nvapi_QueryInterface"));
    check(query, "phoenix dispatch observer query missing");
    auto* target = query(0x24973538);
    check(target && hook(target, reinterpret_cast<void*>(observe_dispatch),
                         reinterpret_cast<void**>(&addon_state.original_dispatch)),
          "phoenix dispatch observer hook failed");
}

std::string sha_file(const wchar_t* path) {
    return phoenix_model::mapped_file_digest(std::filesystem::path(path));
}

// Installed from the first registered queue, before any feature exists, so
// that submission ownership of engine command lists is already observed when
// the first claimable frame arrives.
void install_submission_observer(ID3D12CommandQueue* queue) {
    auto& addon_state = state();
    if (!queue)
        return;
    auto* target = (*reinterpret_cast<void***>(queue))[10];
    if (!addon_state.original_execute) {
        check(hook(target, reinterpret_cast<void*>(execute),
                   reinterpret_cast<void**>(&addon_state.original_execute)),
              "phoenix submit hook failed");
        addon_state.execute_target = target;
    }
    check(target == addon_state.execute_target, "phoenix queue implementation changed");
}
// Identify the supported DLL and explicitly approved variants. Whole-file
// identity is checked alongside entry signatures before installing hooks.
// Always report the observed hash so reference and variant runs are distinct.
inline const char* reference_native_dll_sha256() {
    return "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e";
}
inline bool listed_native_dll(const std::string& allowed, const std::string& observed) {
    if (observed.empty())
        return false;
    size_t cursor = 0;
    while (cursor < allowed.size()) {
        auto end = allowed.find_first_of(",; "
                                         "\t\r\n",
                                         cursor);
        if (end == std::string::npos)
            end = allowed.size();
        std::string one = allowed.substr(cursor, end - cursor);
        if (one.size() == observed.size()) {
            bool same = true;
            for (size_t iteration_index = 0; iteration_index < one.size(); ++iteration_index)
                if (std::tolower(static_cast<unsigned char>(one[iteration_index])) !=
                    std::tolower(static_cast<unsigned char>(observed[iteration_index]))) {
                    same = false;
                    break;
                }
            if (same)
                return true;
        }
        cursor = end + 1;
    }
    return false;
}
void preload_nr_runtime() {
    // Gated on renodx's absence. One attempt per process: a failing
    // LoadLibrary walks the DLL search path every present otherwise.
    // Used by the native path; the direct stage loads by absolute path.
    static std::string tried_folder;
    static bool tried = false;
    const auto& folder = state().panel.nvidia_dll_folder;
    if ((tried && tried_folder == folder) || GetModuleHandleW(L"nvngx_dlssnr.dll"))
        return;
    if (GetModuleHandleW(L"renodx-dlss5.addon64"))
        return;
    tried = true;
    tried_folder = folder;
    const auto dll = phoenix_model::find_nvidia_dll({}, phoenix_model_tools::local_library_directory(),
                                                   std::filesystem::u8path(folder));
    if (!dll.empty() && LoadLibraryW(dll.c_str()))
        reshade::log::message(reshade::log::level::info,
                              "[Phoenix DLSS5] NR runtime pre-loaded (nvngx_dlssnr.dll)");
}
void install_native(ID3D12CommandQueue* queue) {
    auto& addon_state = state();
    install_submission_observer(queue);
    install_ngx_exports();
    install_sl_hooks();
    if (addon_state.direct_game)
        return;
    preload_nr_runtime();
    if (addon_state.original_evaluate)
        return;
    auto module = GetModuleHandleW(L"nvngx_dlssnr.dll");
    // The runtime may still be absent (nothing loaded it yet): export
    // observation above already runs, and the internal hooks install once
    // it appears. Deferral, not refusal -- there is nothing to hook yet.
    if (!module)
        return;
    wchar_t path[32768]{};
    const auto length = GetModuleFileNameW(module, path, 32768);
    check(length && length < 32768, "phoenix cannot identify the native DLL");
    addon_state.native_dll_sha256 = sha_file(path);
    addon_state.native_dll_is_reference =
        addon_state.native_dll_sha256 == reference_native_dll_sha256();
    if (!addon_state.native_dll_is_reference) {
        check(
            listed_native_dll(addon_state.allowed_native_dll, addon_state.native_dll_sha256),
            "phoenix unsupported native DLL identity; list its sha256 in allow_native_dll_sha256 to accept a variant");
        check(addon_state.native_dll_variant_approval ==
                  "I accept driving a native DLL variant with no byte-exactness oracle",
              "phoenix native DLL variant needs its explicit written approval");
    }
    auto* target = reinterpret_cast<unsigned char*>(module) + 0x18620;
    const unsigned char signature[]{0x48, 0x8b, 0xc4, 0x55, 0x53, 0x56, 0x57, 0x41,
                                    0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57, 0x48};
    check(!memcmp(target, signature, sizeof(signature)), "phoenix Evaluate entry changed");
    HMODULE pin = nullptr;
    check(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                             reinterpret_cast<LPCWSTR>(&CnrPHOENIXRegister), &pin) != FALSE,
          "phoenix sidecar pin failed");
    check(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                             reinterpret_cast<LPCWSTR>(target), &pin) != FALSE,
          "phoenix native creation DLL pin failed");
    // D3D12 ReleaseFeature 0x16040 calls common release 0x14760 with RCX
    // unchanged (0x16103..0x16106). Hook inside the caller-module check.
    auto* release_target = reinterpret_cast<unsigned char*>(module) + 0x14760;
    const unsigned char release_signature[]{0x48, 0x89, 0x5c, 0x24, 0x08,
                                            0x48, 0x89, 0x6c, 0x24, 0x18};
    if (!addon_state.original_release) {
        check(!memcmp(release_target, release_signature, sizeof(release_signature)),
              "phoenix release entry changed");
        check(hook(release_target, reinterpret_cast<void*>(release),
                   reinterpret_cast<void**>(&addon_state.original_release)),
              "phoenix release hook failed");
    }
    // Export 0x15750 forwards (instance, cmd, params, outHandle) to 0x17e20
    // at 0x15949..0x15952. Observe success, then consume freshness exactly once.
    if (!addon_state.original_create) {
        auto* create_target = reinterpret_cast<unsigned char*>(module) + 0x17e20;
        const unsigned char create_signature[]{0x40, 0x55, 0x53, 0x56, 0x57,
                                               0x41, 0x54, 0x41, 0x55};
        check(!memcmp(create_target, create_signature, sizeof(create_signature)),
              "phoenix create entry changed");
        check(hook(create_target, reinterpret_cast<void*>(create),
                   reinterpret_cast<void**>(&addon_state.original_create)),
              "phoenix create hook failed");
    }
    check(hook(target, reinterpret_cast<void*>(evaluate),
               reinterpret_cast<void**>(&addon_state.original_evaluate)),
          "phoenix Evaluate hook failed");
    addon_state.evaluate_target = target;
}

// Compare canonical path components case-insensitively. A textual prefix
// would incorrectly accept a sibling such as folder_backup as a child.
bool beneath(const std::filesystem::path& child, const std::filesystem::path& parent) {
    const auto canonical_child = std::filesystem::weakly_canonical(child),
               canonical_parent = std::filesystem::weakly_canonical(parent);
    auto child_component = canonical_child.begin(), parent_component = canonical_parent.begin();
    for (; parent_component != canonical_parent.end(); ++parent_component, ++child_component)
        if (child_component == canonical_child.end() ||
            _wcsicmp(child_component->c_str(), parent_component->c_str()) != 0)
            return false;
    return child_component != canonical_child.end();
}
void configure() {
    auto& addon_state = state();
    if (addon_state.configured)
        return;
    addon_state.configured = true;
#if CNR_PHOENIX_DIAGNOSTIC
    wchar_t path[32768]{};
    const auto config_path_length =
        GetEnvironmentVariableW(L"CNR_PHOENIX_DIAGNOSTIC_CONFIG", path, 32768);
#else
    wchar_t path[32768]{};
    const auto config_path_length =
        GetEnvironmentVariableW(L"CNR_PHOENIX_GAME_CONFIG", path, 32768);
#endif
    if (config_path_length >= 32768)
        throw std::runtime_error("Phoenix DLSS5 configuration path is too long");
    if (!config_path_length) {
        wchar_t module_path[32768]{};
        check(GetModuleFileNameW(addon_state.self, module_path, 32768) != 0,
              "Phoenix DLSS5 addon location unavailable");
        const auto sidecar = std::filesystem::path(module_path).parent_path() / "Phoenix.json";
        if (!std::filesystem::is_regular_file(sidecar)) {
            reshade::log::message(
                reshade::log::level::warning,
                "[Phoenix DLSS5] Not armed: no Phoenix.json beside addon and no launch configuration. Panel alone does not run inference.");
            return;
        }
        check(wcscpy_s(path, sidecar.c_str()) == 0, "Phoenix DLSS5 sidecar path too long");
    }
    auto config = phoenix_bundle::json_file(path);
    if (!addon_state.live_loaded)
        addon_state.panel.nvidia_dll_folder = config.value("nvidia_dll_folder", std::string());
    if (!addon_state.live_loaded)
        addon_state.panel.model_library_directory = config.value("model_library_directory", std::string());
    addon_state.embedded = phoenix_embedded::install(addon_state.self);
    if (!addon_state.embedded.root.empty()) {
        config["bundle"] = phoenix_model::source_path_string(addon_state.embedded.native);
        config["portable_cpp"] = {{"runtime_root", phoenix_model::source_path_string(addon_state.embedded.runtime)}};
        phoenix::portable::prepare_embedded_backend() =
            [module = addon_state.self, cache = addon_state.embedded.root.parent_path()](
                const std::string& backend) { phoenix_embedded::install(module, cache, backend); };
    }
    // Retain the requested path for the panel even if a later configuration
    // check refuses to arm inference. A disarmed configuration is visible too.
    const auto model_source = config.value("weight_source", Json::object());
    if (model_source.value("kind", std::string()) == "nvidia_dll")
        addon_state.model_dll_path =
            std::filesystem::u8path(model_source.value("path", std::string()));
    if (config.contains("portable_worker"))
        addon_state.portable_model_dll_path =
            std::filesystem::u8path(config.at("portable_worker").value("weights", std::string()));
    const auto schema = config.at("schema").get<std::string>();
    addon_state.measure = schema == "production-game-phoenix-measure/1";
    addon_state.interactive = schema == "phoenix-live/1";
    addon_state.observe = schema == "production-game-phoenix-observe/1";
#if CNR_PHOENIX_DIAGNOSTIC
    check(addon_state.measure || schema == "production-game-phoenix-diagnostic/1",
          "phoenix diagnostic configuration schema");
#else
    check(addon_state.interactive || addon_state.measure || addon_state.observe ||
              schema == "production-game-phoenix/2",
          "phoenix configuration schema");
#endif
    if (!config.at("armed").get<bool>())
        return;
    const auto input_provider = config.value("input_provider", std::string("native_nr"));
    check(input_provider == "native_nr" || input_provider == "game_dlss" ||
              input_provider == "feeder_d3d12", "Unknown Phoenix DLSS5 input provider");
    addon_state.direct_feeder = input_provider == "feeder_d3d12";
    addon_state.direct_game = input_provider == "game_dlss" || addon_state.direct_feeder;
    if (addon_state.direct_feeder)
        addon_state.direct_status = "Waiting for a direct feeder texture (no SR/DLAA pass)";
    menu_helper::configure(addon_state.direct_game && config.value("test_menu_navigation", false));
    addon_state.direct_capture =
        addon_state.direct_game && config.value("capture_present_once", false);
    {
        const auto capture_frame = config.value("capture_frame", 1200u);
        addon_state.direct_capture_frame =
            (capture_frame >= 120 && capture_frame <= 4000000) ? capture_frame : 1200;
        addon_state.next_capture_at = addon_state.direct_capture_frame;
        const auto capture_interval = config.value("capture_every", 0u);
        addon_state.capture_every = (capture_interval <= 4000000) ? capture_interval : 0;
    }
    const auto transfer = config.value("direct_transfer", std::string("linear"));
    check(transfer == "linear" || transfer == "srgb", "Unknown direct model color transfer");
    addon_state.direct_srgb = transfer == "srgb";
    check(!addon_state.direct_game || (addon_state.interactive && !CNR_PHOENIX_DIAGNOSTIC),
          "Direct game provider requires live candidate build");
    addon_state.sandbox = std::filesystem::u8path(config.at("sandbox").get<std::string>());
    wchar_t process[32768]{};
    check(GetModuleFileNameW(nullptr, process, 32768) != 0,
          "phoenix cannot identify the running process");
    // By default a run must live in an owned sandbox copy, which is what keeps
    // an original installation read-only. Driving a game in place is a
    // deliberate, separate decision: the configuration has to name that exact
    // directory and say so in as many words, the running executable has to be
    // inside it, and it must not be the workspace's own sandbox root. Nothing
    // is inferred - an absent field means the ordinary sandbox rule applies.
    addon_state.allowed_native_dll = config.value("allow_native_dll_sha256", std::string());
    addon_state.native_dll_variant_approval =
        config.value("native_dll_variant_approval", std::string());
    const auto source_workspace =
        std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    addon_state.in_place = config.value("in_place_game_directory", std::string());
    if (!addon_state.in_place.empty()) {
        check(config.value("in_place_approval", std::string()) ==
                  "I approve driving this installed game in place",
              "phoenix in-place game directory needs its explicit written approval");
        const auto directory = std::filesystem::u8path(addon_state.in_place);
        check(std::filesystem::is_directory(directory),
              "phoenix in-place game directory does not exist");
        check(std::filesystem::weakly_canonical(directory) ==
                  std::filesystem::weakly_canonical(addon_state.sandbox),
              "phoenix in-place game directory must be the configured sandbox");
        check(!beneath(addon_state.sandbox, source_workspace),
              "phoenix in-place mode is for an installed game, not a workspace sandbox");
        check(beneath(process, addon_state.sandbox),
              "phoenix running executable is not inside the named in-place directory");
    } else {
        check(beneath(addon_state.sandbox, source_workspace / "sandboxes") ||
                  beneath(addon_state.sandbox, source_workspace / "build"),
              "Phoenix DLSS5 sandbox outside this build workspace");
        check(beneath(process, addon_state.sandbox),
              "phoenix original installed game is read-only; use owned sandbox");
    }
    if (config.contains("portable_worker")) {
        const auto& worker = config.at("portable_worker");
        addon_state.worker_python = std::filesystem::u8path(worker.value("python", std::string()));
        addon_state.worker_script = std::filesystem::u8path(worker.value("script", std::string()));
        addon_state.worker_weights =
            std::filesystem::u8path(worker.value("weights", std::string()));
        addon_state.worker_vulkan_build =
            std::filesystem::u8path(worker.value("vulkan_build", std::string()));
        addon_state.worker_hip_build =
            std::filesystem::u8path(worker.value("hip_build", std::string()));
        addon_state.worker_sycl_build =
            std::filesystem::u8path(worker.value("sycl_build", std::string()));
        const auto engine = worker.value("vulkan_engine", std::string("auto"));
        check(engine == "auto" || engine == "portable" || engine == "matrix" || engine == "fp8",
              "Unknown Vulkan engine");
        addon_state.panel.worker_engine = engine;
    }
    if (config.contains("portable_cpp"))
        phoenix::portable::configure_native_runtime(std::filesystem::u8path(
            config.at("portable_cpp").at("runtime_root").get<std::string>()));
    addon_state.worker_vulkan_build = phoenix::portable::native_runtime_root() / "vulkan";
    addon_state.execution_devices =
        phoenix::execution_devices(addon_state.worker_vulkan_build, {}, {});
    phoenix::portable::append_cpp_devices(addon_state.execution_devices);
    addon_state.output_directory = addon_state.sandbox / "PhoenixOutput";
    std::filesystem::create_directories(addon_state.output_directory);
    if (!addon_state.observe) {
        const auto bundle = config.value("bundle", std::string());
        check(
            !bundle.empty() || (addon_state.direct_game && (config.contains("portable_cpp") ||
                                                            config.contains("portable_worker"))),
            "Native execution needs a model bundle; portable execution needs a compiled runtime configuration");
        addon_state.bundle_path = std::filesystem::u8path(bundle);
        addon_state.configured_weight_source =
            phoenix_model::WeightSource::config(config.value("weight_source", Json::object()));
        addon_state.weight_source = addon_state.configured_weight_source;
        if (addon_state.live_loaded && phoenix::uses_ours(addon_state.panel.mode))
            addon_state.weight_source = phoenix_model_tools::selection(addon_state.panel, addon_state.configured_weight_source);
        addon_state.active_model_source = addon_state.panel.model_source;
        addon_state.active_model_folder = addon_state.panel.model_folder;
        addon_state.active_model_dll = addon_state.panel.model_dll;
        addon_state.active_model_id = addon_state.panel.model_id;
        addon_state.active_nvidia_dll_folder = addon_state.panel.nvidia_dll_folder;
    }
    // Marker-based control is opt-in for production and retained for diagnostic runs.
    addon_state.require_gpu_ready_file =
        config.value("require_gpu_ready_file", CNR_PHOENIX_DIAGNOSTIC != 0);
#if CNR_PHOENIX_DIAGNOSTIC
    addon_state.gate = addon_state.sandbox / "production_game_phoenix_diagnostic_gpu.ready";
#else
    addon_state.gate = addon_state.sandbox / "production_game_phoenix_gpu.ready";
#endif
    if (!addon_state.observe)
        addon_state.frame_limit = config.at("frame_limit").get<unsigned>();
    addon_state.routing_test_period = config.value("routing_test_period", 0u);
    check(!addon_state.routing_test_period ||
              (!CNR_PHOENIX_DIAGNOSTIC && addon_state.routing_test_period <= 32 &&
               addon_state.frame_limit <= 512),
          "Phoenix DLSS5 invalid bounded routing test");
    if (addon_state.measure) {
        // A timed window needs more than the first-live budget, so the
        // measurement schema carries its own explicit, still bounded cap.
        // A multi-hour soak needs a frame budget that a multi-hour soak can
        // actually reach. Still an explicit bound, just a much larger one.
        check(addon_state.frame_limit >= 1 && addon_state.frame_limit <= 4000000,
              "phoenix measured limit must be 1..4000000");
        addon_state.claim_enabled = config.at("claim").get<bool>();
        addon_state.warmup_ms = config.at("warmup_seconds").get<uint64_t>() * 1000;
        addon_state.measure_ms = config.at("measure_seconds").get<uint64_t>() * 1000;
        check(addon_state.measure_ms >= 1000 && addon_state.measure_ms <= 14400000,
              "phoenix measured window must be 1..14400 seconds");
        check(addon_state.warmup_ms <= 300000,
              "phoenix measured warmup must be at most 300 seconds");
    } else if (addon_state.interactive) {
        check(addon_state.frame_limit >= 1 && addon_state.frame_limit <= 4000000,
              "Phoenix DLSS5 live frame safety cap");
        addon_state.capture_limit = 0;
    } else if (addon_state.observe) {
        // Record-only: never claims, tests, skips, or exceeds the record cap.
        // Past the cap the detour keeps forwarding silently (see evaluate).
        // No color_domain declaration is read here: observation must never
        // depend on an unproven SDR-proxy claim.
        check(phoenix::observe_config_valid(config.value("claim", false),
                                            config.contains("routing_test_period"),
                                            config.value("routing_test_period", 0u),
                                            !config.value("skip_groups", std::string()).empty(),
                                            config.value("frame_limit", 0u)),
              "Phoenix DLSS5 observe configuration rejected");
        addon_state.frame_limit = config.value("frame_limit", 0u);
        addon_state.capture_limit = 0;
        addon_state.claim_enabled = false;
    } else {
        // The first-live gate was deliberately capped at 32 pairs. Correctness
        // over a longer run is a separate question from the first gate, so the
        // bounded path now carries its own larger but still bounded cap, and a
        // separate budget for how many raw captures reach the disk.
        check(addon_state.frame_limit >= 1 && addon_state.frame_limit <= 4096,
              "phoenix bounded limit must be 1..4096");
        addon_state.capture_limit = config.value("capture_limit", 32u);
        check(addon_state.capture_limit >= 1 &&
                  addon_state.capture_limit <= addon_state.frame_limit,
              "phoenix capture budget must be 1..frame_limit");
    }
    const auto skip_groups = config.value("skip_groups", std::string());
    check(skip_groups.empty(),
          "Phoenix DLSS5 release does not permit partial-network stage skipping");
    skip_groups_mask.store(0, std::memory_order_relaxed);
    // Production uses chained natural groups and the complete barrier contract.
    addon_state.group_batched.assign(15, 1);
    if (!addon_state.observe)
        addon_state.sdr_proxy = config.at("color_domain") == "game_nr_sdr_proxy";
    if (!addon_state.observe)
        addon_state.hdr_bridge = config.at("color_domain") == "game_hdr_bridge";
    addon_state.hdr_control_override =
        addon_state.hdr_bridge && config.value("hdr_control_override", false);
    check(
        addon_state.observe || addon_state.sdr_proxy || addon_state.hdr_bridge,
        "phoenix direct NR path requires the game's existing SDR proxy or the HDR bridge; no double HDR conversion");
    HMODULE pin = nullptr;
    check(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                             reinterpret_cast<LPCWSTR>(&CnrPHOENIXRegister), &pin) != FALSE,
          "phoenix observer sidecar pin failed");
    addon_state.automatic = true;
    reshade::log::message(
        reshade::log::level::info,
        addon_state.direct_feeder
            ? "[Phoenix DLSS5] Armed: direct feeder textures, own network, no SR/DLAA feature"
        : addon_state.direct_game
            ? "[Phoenix DLSS5] Armed: direct game SR/RR input, own network, no NR feature/provider required"
            : "[Phoenix DLSS5] Armed: native NR boundary provider");
}
bool gpu_gate() noexcept {
    auto& addon_state = state();
    return phoenix::direct::recording_allowed(
        addon_state.automatic, addon_state.require_gpu_ready_file, [&] {
            std::error_code ec;
            return std::filesystem::is_regular_file(addon_state.gate, ec) && !ec;
        });
}
// The runtime's documented escape hatch for retrieving the object a proxy
// wraps, queried rather than assumed. An answer is accepted only when it is
// independently confirmed and names a list the runtime registered with us.
constexpr GUID kUnwrappedObject = {
    0x7f2c9a11, 0x3b4e, 0x4d6a, {0x81, 0x2f, 0x5e, 0x9c, 0xd3, 0x7a, 0x1b, 0x42}};
constexpr GUID kIdentityCookie = {
    0xc4e17a55, 0x9f3b, 0x4c2e, {0xb6, 0xd1, 0x2f, 0x08, 0xa7, 0x31, 0x5e, 0x90}};

bool observed_list(ID3D12GraphicsCommandList* cmd) {
    auto& observer_state = observation::observer();
    std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
    return observer_state.commands.count(cmd) != 0;
}
// Confirms, without relying on any wrapper contract, that the supplied pointer
// and the candidate are the same underlying D3D12 object: private data written
// through one is observable through it and through no other registered list.
bool same_object(ID3D12GraphicsCommandList* supplied, ID3D12GraphicsCommandList* candidate) {
    static uint64_t counter = 0;
    const uint64_t cookie = 0x594931434f4f4bull + ++counter;
    if (FAILED(supplied->SetPrivateData(kIdentityCookie, sizeof(cookie), &cookie)))
        return false;
    unsigned matches = 0;
    bool hit = false;
    {
        auto& observer_state = observation::observer();
        std::lock_guard<std::recursive_mutex> lock(observer_state.mutex);
        for (const auto& entry : observer_state.commands) {
            uint64_t value = 0;
            UINT size = sizeof(value);
            if (SUCCEEDED(entry.first->GetPrivateData(kIdentityCookie, &size, &value)) &&
                size == sizeof(value) && value == cookie) {
                ++matches;
                if (entry.first == candidate)
                    hit = true;
            }
        }
    }
    supplied->SetPrivateData(kIdentityCookie, 0, nullptr); // leave no residue
    return matches == 1 && hit;
}
// The engine hands the NR boundary whatever pointer it holds, which under this
// runtime is a proxy rather than the original command list. Recording, state
// observation and submission all have to name the same object.
ID3D12GraphicsCommandList* resolve_native_list(ID3D12GraphicsCommandList* supplied) {
    auto& addon_state = state();
    check(supplied != nullptr, "phoenix null Evaluate command list");
    if (observed_list(supplied))
        return supplied;
    const auto cached = addon_state.originals.find(supplied);
    ComPtr<IUnknown> unwrapped;
    check(SUCCEEDED(supplied->QueryInterface(kUnwrappedObject,
                                             reinterpret_cast<void**>(unwrapped.GetAddressOf()))) &&
              unwrapped,
          "phoenix Evaluate list is neither an observed original nor an unwrappable proxy");
    auto* candidate = reinterpret_cast<ID3D12GraphicsCommandList*>(unwrapped.Get());
    if (cached != addon_state.originals.end()) {
        check(cached->second == candidate, "phoenix wrapped Evaluate list changed identity");
        return candidate;
    }
    check(observed_list(candidate),
          "phoenix unwrapped Evaluate list was never registered by the runtime");
    check(candidate->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT,
          "phoenix unwrapped Evaluate list is not DIRECT");
    check(supplied->GetType() == candidate->GetType(),
          "phoenix unwrapped Evaluate list changed command list type");
    // A wrapped list reports the wrapped device, so the two device pointers are
    // expected to differ by exactly the same wrapping. Require that they agree
    // either directly or after the device is unwrapped the same documented way.
    ComPtr<ID3D12Device> supplied_device, candidate_device;
    phoenix_native::hr(supplied->GetDevice(IID_PPV_ARGS(&supplied_device)));
    phoenix_native::hr(candidate->GetDevice(IID_PPV_ARGS(&candidate_device)));
    bool same_device = supplied_device.Get() == candidate_device.Get();
    if (!same_device) {
        ComPtr<IUnknown> original_device;
        if (SUCCEEDED(supplied_device->QueryInterface(
                kUnwrappedObject, reinterpret_cast<void**>(original_device.GetAddressOf()))) &&
            original_device)
            same_device = static_cast<void*>(original_device.Get()) ==
                          static_cast<void*>(candidate_device.Get());
    }
    check(same_device, "phoenix unwrapped Evaluate list belongs to another device");
    check(same_object(supplied, candidate),
          "phoenix unwrap not confirmed by independent object identity");
    addon_state.originals.emplace(supplied, candidate);
    ++addon_state.unwrapped_lists;
    return candidate;
}
bool registered_queue(ID3D12CommandQueue* queue) {
    for (auto* known : state().queues)
        if (reinterpret_cast<ID3D12CommandQueue*>(known->get_native()) == queue)
            return true;
    return false;
}
// The submitting queue is established from this exact list's own observed
// submissions, never from "some DIRECT queue" and never from the last queue
// seen. It is checked again against the real submission before Execute runs.
ID3D12CommandQueue* resolve_queue(ID3D12GraphicsCommandList* cmd) {
    auto& addon_state = state();
    check(addon_state.original_execute != nullptr,
          "phoenix submission observation was not installed before the claimed frame");
    const auto found = addon_state.ownership.find(cmd);
    check(found != addon_state.ownership.end() && found->second.submissions > 0 &&
              !found->second.ambiguous,
          "phoenix Evaluate list has no single observed submission queue yet");
    auto* queue = found->second.queue;
    check(queue && registered_queue(queue),
          "phoenix submitting queue was never registered or was destroyed");
    check(queue && queue->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT,
          "phoenix Evaluate list is submitted on a non-DIRECT queue");
    ComPtr<ID3D12Device> report_rows, b;
    phoenix_native::hr(cmd->GetDevice(IID_PPV_ARGS(&report_rows)));
    phoenix_native::hr(queue->GetDevice(IID_PPV_ARGS(&b)));
    check(report_rows == b, "phoenix submitting queue belongs to another device");
    return queue;
}
// Establishes the engine output state from evidence only. An engine may leave
// the resource untransitioned in this recording, so queue-ordered history and
// an untouched creation state are both admissible; a guess is not.
void resolve_engine_output(ID3D12GraphicsCommandList* cmd, ID3D12CommandQueue* queue,
                           const NVSDK_NGX_Parameter* params, FrameEvidence& frame) {
    auto& addon_state = state();
    addon_state.output_state_from_committed_observation = false;
    auto* output = parameter_resource(params, "DLSSNR.Output", true);
    D3D12_RESOURCE_STATES observed{};
    if (observation::epoch_state(cmd, output, observed)) {
        frame.output_state = observed;
        frame.output_state_verified = true;
        addon_state.output_state_basis = "transition_in_current_recording";
        return;
    }
    check(observation::permits_ledger_fallback(cmd),
          "phoenix engine output state requires a completely observed recording epoch");
    auto answer = addon_state.states.state(uint64_t(output), uint64_t(cmd), uint64_t(queue));
    // ReShade's init_resource does not reach us for a device created before
    // ReShade loaded, so the ledger can be empty for a resource that is
    // nevertheless perfectly well understood. Our own CreateCommittedResource
    // observation covers exactly that gap, and only for a private committed
    // texture created in UNORDERED_ACCESS that no barrier has ever named.
    D3D12_RESOURCE_STATES created{};
    if (!answer.known && observation::committed_creation_state(output, created)) {
        answer.known = true;
        answer.state = uint32_t(created);
        answer.basis = ledger::Basis::creation;
        addon_state.output_state_from_committed_observation = true;
    }
    if (!answer.known) {
        try {
            addon_state.output_state_refusal = observation::committed_detail(output) + " " +
                                               observation::aliasing_detail(cmd, output);
        } catch (...) {
        }
        try {
            const auto accounted =
                addon_state.states.explain(uint64_t(output), uint64_t(cmd), uint64_t(queue));
            const auto bit = [](bool enabled) { return enabled ? "1" : "0"; };
            addon_state.output_state_refusal +=
                " ledger(tracked=" + std::string(bit(accounted.tracked)) +
                " open_here=" + bit(accounted.open_in_this_list) +
                " others=" + std::to_string(accounted.other_open_recordings) +
                " committed=" + bit(accounted.committed) +
                " committed_queue_matches=" + bit(accounted.committed_queue_matches) +
                " created_never_transitioned=" + bit(accounted.created_never_transitioned) +
                " transitions=" + std::to_string(accounted.transitions) +
                " open_lists=" + std::to_string(accounted.open_recording_lists) + ")";
            D3D12_RESOURCE_STATES discarded{};
            if (observation::discarded_epoch_state(cmd, output, discarded))
                addon_state.output_state_refusal +=
                    " discarded_in_this_list=" + std::to_string(unsigned(discarded));
            else
                addon_state.output_state_refusal += " discarded_in_this_list=absent";
        } catch (...) {
        }
        check(
            false,
            "phoenix engine output state is not established by any observed transition or creation state");
    }
    if (!observation::aliasing_permits(cmd, output)) {
        try {
            addon_state.output_state_refusal =
                observation::aliasing_detail(cmd, output) +
                " ledger_basis=" + std::string(ledger::basis_name(answer.basis));
        } catch (...) {
        }
        check(false,
              "phoenix engine output was named by an aliasing barrier in this recording epoch");
    }
    frame.output_state = D3D12_RESOURCE_STATES(answer.state);
    frame.output_state_verified = true;
    addon_state.output_state_basis = addon_state.output_state_from_committed_observation
                                         ? "observed_private_committed_creation_never_barriered"
                                         : ledger::basis_name(answer.basis);
}
bool ownership_established(ID3D12GraphicsCommandList* cmd) noexcept {
    try {
        return resolve_queue(cmd) != nullptr;
    } catch (const std::exception& feature_entry) {
        try {
            state().ownership_refusal = feature_entry.what();
        } catch (...) {
        }
        return false;
    } catch (...) {
        return false;
    }
}
bool resolve_automatic(void*, ID3D12GraphicsCommandList* cmd, const NVSDK_NGX_Handle*,
                       const NVSDK_NGX_Parameter* params, FrameEvidence* frame,
                       bool require_output) {
    frame->queue = resolve_queue(cmd);
    frame->color_is_nr_sdr_proxy = state().sdr_proxy;
    frame->binding_lease = observation::resolve(cmd, params, *frame, false);
    if (require_output)
        resolve_engine_output(cmd, frame->queue, params, *frame);
    return true;
}
bool resolve_automatic(void* context, ID3D12GraphicsCommandList* cmd,
                       const NVSDK_NGX_Handle* feature, const NVSDK_NGX_Parameter* params,
                       FrameEvidence* frame) {
    return resolve_automatic(context, cmd, feature, params, frame, true);
}
void prepare_automatic(ID3D12GraphicsCommandList* cmd, const NVSDK_NGX_Handle* feature,
                       const NVSDK_NGX_Parameter* params) {
    auto& addon_state = state();
    check(params, "phoenix null parameters");
    auto* queue = resolve_queue(cmd);
    auto* color = parameter_resource(params, "DLSSNR.Color", true);
    const auto desc = color->GetDesc();
    if (addon_state.endpoint.session && !addon_state.endpoint.session->matches(queue, desc)) {
        int reset = 0;
        check(NVSDK_NGX_SUCCEED(params->Get("DLSSNR.Reset", &reset)) && reset,
              "phoenix geometry/queue rebuild requires native Reset");
        check(!addon_state.endpoint.session->command(),
              "phoenix geometry/queue changed before submit");
        retire();
    }
    // A recording-thread change ends the session the same way a geometry or
    // queue change does: the Transaction admits exactly one recorder
    // (CPU-pinned refusal), and sequential frames on different threads are
    // legitimate engine behavior, not a contract breach. Queue and geometry
    // already match here, so no engine Reset is required -- only a rebuild
    // on the new thread with a fresh candidate history.
    if (addon_state.endpoint.session && !addon_state.endpoint.session->recording_thread()) {
        check(!addon_state.endpoint.session->command(),
              "phoenix recording thread changed before submit");
        retire();
        addon_state.cold_restart = true;
    }
    if (addon_state.endpoint.session)
        return;
    // Resolve controls, states and bindings before allocating GPU resources.
    // Claim-time proof is validation-only (nothing reads proof back), so
    // the routine output-state refusal degrades here exactly as it does on
    // the per-frame path: queue/binding attribution still refuses the claim
    // terminally, but an unestablished engine output only withholds native
    // capture (candidate-only pairs). The refusal detail is already filed
    // in s.output_state_refusal by resolve_engine_output.
    FrameEvidence proof;
    resolve_automatic(nullptr, cmd, feature, params, &proof, false);
    try {
        resolve_engine_output(cmd, proof.queue, params, proof);
    } catch (const std::runtime_error&) {
    }
    const auto controls = hdr_overridden_controls(
        effective_controls(
            read_controls(*params,
                          [](NVSDK_NGX_Result tag_row) { return NVSDK_NGX_SUCCEED(tag_row); }),
            true, addon_state.fresh_claim, addon_state.cold_restart),
        addon_state.hdr_control_override);
    auto* motion = parameter_resource(params, "DLSSNR.MVec");
    resolve_controls(controls, unsigned(desc.Width), desc.Height, motion != nullptr, false, 0);
    collect();
    check(addon_state.retired.size() < 8, "phoenix retired generation limit");
    if (!addon_state.bundle) {
        addon_state.bundle = std::make_unique<phoenix_bundle::Bundle>(
            addon_state.bundle_path, [](const auto& file) { return sha_file(file.c_str()); },
            addon_state.weight_source, true);
        check(addon_state.bundle->weights.at("unknown_gap_bytes") == 0,
              "phoenix owned game requires proven complete weights");
        addon_state.candidates =
            phoenix_kernel::load_catalog(phoenix_native::catalog(*addon_state.bundle));
    }
    const Json* selected = &addon_state.bundle->select_frame(
        unsigned(desc.Width), desc.Height, [&](const Json& frame) {
            phoenix_native::preflight(frame, addon_state.candidates);
        });
    progress("plan_selected");
    // Choose the allocation device by evidence: its objects must come back
    // unwrapped, and its allocations must be owned by the same device the
    // engine's own inputs report. Nothing here assumes a layering.
    const auto choice = resolve_allocation_device(queue, color);
    check(choice.resolved,
          "phoenix no device allocates unwrapped objects owned by the engine input device");
    auto device = choice.device;
    progress("device_resolved");
    // Device identity evidence, recorded before any owned allocation so a
    // rejection names the objects involved instead of only a predicate.
    try {
        auto& record = addon_state.device_identity;
        record["queue"] = uint64_t(queue);
        ComPtr<ID3D12Device> reported;
        phoenix_native::hr(queue->GetDevice(IID_PPV_ARGS(&reported)));
        record["queue_reported_device"] = uint64_t(reported.Get());
        record["queue_reported_device_was_a_proxy"] = reported.Get() != device.Get();
        ComPtr<ID3D12Device> list_reported;
        phoenix_native::hr(cmd->GetDevice(IID_PPV_ARGS(&list_reported)));
        record["list_reported_device"] = uint64_t(list_reported.Get());
        record["list_reported_device_was_a_proxy"] = list_reported.Get() != device.Get();
        record["queue_device"] = uint64_t(device.Get());
        record["allocation_device_basis"] = choice.basis;
        record["allocation_device_candidates_tried"] = choice.candidates_tried;
        ComPtr<IUnknown> queue_device_unknown;
        phoenix_native::hr(device.As(&queue_device_unknown));
        record["queue_device_iunknown"] = uint64_t(queue_device_unknown.Get());
        auto probe = phoenix_native::image(device.Get(), 4, 4, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, false);
        ComPtr<ID3D12Device> probe_device;
        phoenix_native::hr(probe->GetDevice(IID_PPV_ARGS(&probe_device)));
        ComPtr<IUnknown> probe_unknown;
        phoenix_native::hr(probe_device.As(&probe_unknown));
        record["owned_image_device"] = uint64_t(probe_device.Get());
        record["owned_image_device_iunknown"] = uint64_t(probe_unknown.Get());
        record["owned_image_device_matches"] = probe_unknown.Get() == queue_device_unknown.Get();
        auto* engine_color = parameter_resource(params, "DLSSNR.Color", true);
        ComPtr<ID3D12Device> engine_device;
        phoenix_native::hr(engine_color->GetDevice(IID_PPV_ARGS(&engine_device)));
        ComPtr<IUnknown> engine_unknown;
        phoenix_native::hr(engine_device.As(&engine_unknown));
        record["engine_color_device"] = uint64_t(engine_device.Get());
        record["engine_color_device_iunknown"] = uint64_t(engine_unknown.Get());
        record["engine_color_device_matches"] = engine_unknown.Get() == queue_device_unknown.Get();
        record["engine_color_format"] = unsigned(desc.Format);
        record["engine_motion_format"] =
            unsigned(motion ? motion->GetDesc().Format : DXGI_FORMAT_UNKNOWN);
        record["plan_width"] = selected->at("width");
        record["plan_height"] = selected->at("height");
    } catch (const std::exception& feature_entry) {
        addon_state.device_identity["evidence_error"] = feature_entry.what();
    }
    progress("installing_dispatch_audit");
    install_dispatch_audit();
    progress("constructing_session");
    auto allocation_generation =
        addon_state.owned_allocations.begin(unsigned(desc.Width), desc.Height);
    // Cold restart rides the native-fresh slot: our candidate history really
    // is empty on a rebuilt session; the engine side is untouched (its reset
    // still comes only from its own parameters). Both Session record gates
    // admit the first frame on this basis, exactly as at feature birth.
    const auto packed_weights = addon_state.bundle->load_weight_view();
    auto session = std::make_shared<Session>(
        device.Get(), queue, *selected, packed_weights, addon_state.candidates, desc.Format,
        motion ? motion->GetDesc().Format : DXGI_FORMAT_R32G32_FLOAT,
        addon_state.fresh_claim || addon_state.cold_restart, allocation_generation,
        addon_state.hdr_bridge, addon_state.hdr_control_override);
    addon_state.fresh_claim = false;
    addon_state.cold_restart = false;
    if (addon_state.measure && !addon_state.timestamps) {
        addon_state.timestamps = std::make_unique<timing::Timestamps>(device.Get(), queue);
        // Group boundaries are marked from inside the recorder, so the
        // breakdown covers exactly the dispatches it already issues.
    }
#if !CNR_PHOENIX_DIAGNOSTIC
    if (!addon_state.live_timing || !addon_state.live_timing->matches(queue)) {
        check(addon_state.retired_timing.size() < 8, "Phoenix DLSS5 timing retirement bound");
        if (addon_state.live_timing) {
            addon_state.live_timing->budget(nullptr);
            addon_state.retired_timing.push_back(std::move(addon_state.live_timing));
        }
        addon_state.live_timing =
            std::make_unique<phoenix::GpuTiming>(device.Get(), queue, addon_state.timing_archive);
        addon_state.live_timing->budget(&addon_state.depth_budget);
        addon_state.timing_settings_dirty = true;
        addon_state.depth_budget.reset_samples();
    }
#endif
    phoenix_native::group_marker = [](ID3D12GraphicsCommandList* list, unsigned group) {
        auto& inner = state();
        if (inner.timestamps && inner.timestamps->recording())
            inner.timestamps->group(list);
        if (inner.live_timing)
            inner.live_timing->group(list, group);
    };
    Json geometry;
    geometry["width"] = desc.Width;
    geometry["height"] = desc.Height;
    geometry["color_format"] = unsigned(desc.Format);
    geometry["hdr_bridge"] = addon_state.hdr_bridge;
    geometry["hdr_control_override"] = addon_state.hdr_control_override;
    geometry["recording_thread"] =
        uint64_t(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    record_event("session_constructed", std::move(geometry));
    progress("session_constructed");
    install_native(queue);
    addon_state.endpoint = {std::move(session), feature,
                            static_cast<ResolveFrame>(resolve_automatic), nullptr};
    ++addon_state.generations;
}
static const char* phoenix_state_name(D3D12_RESOURCE_STATES state) {
    switch (state) {
    case D3D12_RESOURCE_STATE_COMMON:
        return "COMMON/PRESENT";
    case D3D12_RESOURCE_STATE_COPY_DEST:
        return "COPY_DEST";
    case D3D12_RESOURCE_STATE_COPY_SOURCE:
        return "COPY_SOURCE";
    case D3D12_RESOURCE_STATE_UNORDERED_ACCESS:
        return "UAV";
    case D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:
        return "NON_PS_SRV";
    case D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE:
        return "PS_SRV";
    case D3D12_RESOURCE_STATE_RENDER_TARGET:
        return "RTV";
    default:
        return "OTHER";
    }
}
// Crash-safe crumb: the ReShade log can lose its tail in a hard crash, so
// the death-point trail goes to its own file, flushed and closed per write.
// Verbose diagnostics only: synchronous per-frame file IO otherwise throttles
// the CPU, especially the native route, without appearing in GPU timestamps.
static void phoenix_crumb(const char* text) {
    if (state().panel.log_level != 2)
        return;
    char path[MAX_PATH];
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    if (char* slash = strrchr(path, '\\'))
        *(slash + 1) = 0;
    strcat_s(path, "Phoenix DLSS5-crumbs.log");
    FILE* log_file = nullptr;
    if (fopen_s(&log_file, path, "a") == 0 && log_file) {
        fprintf(log_file, "[%llu][%lu] %s\n", GetTickCount64(), (unsigned long)GetCurrentThreadId(),
                text);
        fflush(log_file);
        fclose(log_file);
    }
}
// Crash-forensics breadcrumb: pointer, geometry, format, flags and the state
// we assume. Logged before driver submission so a submit-time death still
// names the exact resources on the list.
static std::string phoenix_describe(ID3D12Resource* resource, D3D12_RESOURCE_STATES assumed) {
    if (!resource)
        return "null";
    const auto description = resource->GetDesc();
    char text[192];
    snprintf(text, sizeof(text),
             "@%llx %llux%llu fmt=%u dim=%u mips=%u array=%u samples=%u flags=0x%x assumed=%s",
             (unsigned long long)resource, (unsigned long long)description.Width,
             (unsigned long long)description.Height, (unsigned)description.Format,
             (unsigned)description.Dimension, (unsigned)description.MipLevels,
             (unsigned)description.DepthOrArraySize, (unsigned)description.SampleDesc.Count,
             (unsigned)description.Flags, phoenix_state_name(assumed));
    return text;
}
void capture_present_guides(ID3D12GraphicsCommandList* supplied, const NVSDK_NGX_Parameter* params) {
    auto& value=state();
    if(!params || value.panel.processing.reuse_guides==0)return;
    auto* cmd=resolve_native_list(supplied);auto* queue=resolve_queue(cmd);
    auto* motion=parameter_resource(params,NVSDK_NGX_Parameter_MotionVectors);
    auto* depth=parameter_resource(params,NVSDK_NGX_Parameter_Depth);
    auto* output=parameter_resource(params,NVSDK_NGX_Parameter_Output);
    D3D12_RESOURCE_STATES ms{},ds{};
    if(!output||!motion||!depth||!observation::epoch_state(cmd,motion,ms)||
       !observation::epoch_state(cmd,depth,ds))return;
    phoenix::direct::Resources resources(output,motion);
    FrameEvidence proof;
    proof.queue=queue;
    proof.binding_lease=observation::resolve(cmd,&resources,proof,false);
    struct RestoreBindings {
        FrameEvidence& proof;
        ID3D12GraphicsCommandList* command;
        ~RestoreBindings() {
            if(proof.restore_bindings)proof.restore_bindings(proof.restore_context,command);
        }
    } restore{proof,cmd};
    if(value.present_guides&&!value.present_guides->owner(queue)) {
        if(!value.present_guides->idle())return;
        value.present_guides.reset();
    }
    if(!value.present_guides) {
        ComPtr<ID3D12Device> device;phoenix_native::hr(cmd->GetDevice(IID_PPV_ARGS(&device)));
        value.present_guides=std::make_unique<phoenix::direct::PresentGuides>(device.Get(),queue);
    }
    float sx=1,sy=1;unsigned flags=0;
    params->Get("MV.Scale.X",&sx);params->Get("MV.Scale.Y",&sy);
    params->Get("DLSS.Feature.Create.Flags",&flags);
    const auto d=output->GetDesc();
    value.present_guides->record(cmd,motion,ms,depth,ds,unsigned(d.Width),d.Height,
                                value.present_serial,sx,sy,(flags&8)!=0);
}
void direct_game_frame(ID3D12GraphicsCommandList* supplied, const NVSDK_NGX_Handle* feature,
                       const NVSDK_NGX_Parameter* params) {
    auto& addon_state = state();
    if (!addon_state.direct_game || !gpu_gate() || addon_state.terminal || !params)
        return;
    ++addon_state.direct_calls;
    bool recording = false;
    bool initializing = false;
    try {
        check((!GetModuleHandleW(L"renodx-dlss5.addon64") && !GetModuleHandleW(L"renodx-dlss.addon64")),
              "Another neural-rendering addon is loaded; disable it to prevent double processing");
        if (addon_state.panel.mode == phoenix::input) {
            addon_state.direct_status = "Game output unchanged; showing the game DLSS input";
            addon_state.last_mode = addon_state.panel.mode;
            addon_state.live.reset_requested = true;
            addon_state.displayed = addon_state.panel;
            addon_state.displayed.mode = phoenix::input;
            addon_state.display_tick = GetTickCount64();
            return;
        }
        if (!addon_state.panel.uplift ||
            (addon_state.panel.processing.enabled && addon_state.panel.processing.hook == 0)) {
            // Master switch off: return BEFORE any GPU timing begin(). The old
            // late return opened a timing slot without end(); the next submit
            // hook then threw "Phoenix DLSS5 timing submission identity" and latched
            // terminal, so re-enabling could never recover. Request a history
            // reset so the re-enable edge starts clean; last_mode is left
            // alone so a mode change while off still triggers reset_hold.
            addon_state.direct_status =
                "Neural rendering off (Uplift disabled); showing the game DLSS input";
            addon_state.live.reset_requested = true;
            addon_state.displayed = addon_state.panel;
            addon_state.displayed.mode = phoenix::input;
            addon_state.display_tick = GetTickCount64();
            return;
        }
        check(addon_state.panel.valid(), "Independent panel settings invalid");
        if (addon_state.panel.execution_device != "native" &&
            phoenix::uses_ours(addon_state.panel.mode) && addon_state.panel.host_unjitter) {
            addon_state.direct_status =
                "Turn off reference-grid unjitter for portable processing; game output preserved";
            addon_state.live.reset_requested = true;
            return;
        }
        check(addon_state.recorded < addon_state.frame_limit, "Direct frame safety cap reached");
        auto* cmd = resolve_native_list(supplied);
        auto* queue = resolve_queue(cmd);
        if (addon_state.panel.mode == phoenix::nvidia) {
            char ctx[160];
            snprintf(ctx, sizeof(ctx), "nr-frame: claim cmd type=%d queue type=%d",
                     cmd ? int(cmd->GetType()) : -1, queue ? int(queue->GetDesc().Type) : -1);
            phoenix_crumb(ctx);
        }
        auto* color = parameter_resource(params, phoenix_render_stage ? NVSDK_NGX_Parameter_Color : NVSDK_NGX_Parameter_Output, true);
        auto* motion = parameter_resource(params, NVSDK_NGX_Parameter_MotionVectors, true);
        const auto color_description = color->GetDesc(), motion_description = motion->GetDesc();
        const bool own_route = phoenix::uses_ours(addon_state.panel.mode);
        const bool native_network = own_route && addon_state.panel.execution_device == "native";
        const auto resolution = addon_state.panel.processing.enabled
            ? phoenix::processing_resolution(unsigned(color_description.Width), color_description.Height,
                                              addon_state.panel.processing, native_network)
            : phoenix::choose_resolution(
            unsigned(color_description.Width), color_description.Height,
            addon_state.panel.resolution_mode, addon_state.panel.resolution_quality,
            addon_state.panel.resolution_aspect, addon_state.panel.resolution_ratio_width,
            addon_state.panel.resolution_ratio_height, native_network);

        const auto describe = [](const D3D12_RESOURCE_DESC& resource_description) {
            return Json{{"width", resource_description.Width},
                        {"height", resource_description.Height},
                        {"format", unsigned(resource_description.Format)},
                        {"dimension", unsigned(resource_description.Dimension)},
                        {"samples", resource_description.SampleDesc.Count},
                        {"flags", unsigned(resource_description.Flags)}};
        };
        addon_state.bridge_inputs = {{"color", describe(color_description)},
                                     {"motion", describe(motion_description)}};
        addon_state.bridge_input_tick = GetTickCount64();
        check(!native_network ||
                  phoenix::direct::extent_supported(resolution.width, resolution.height),
              "Native resolution must be between 1 and 16384 pixels per axis");

        if (!phoenix::direct::source_format(color_description.Format) ||
            !phoenix::direct::motion_format(motion_description.Format))
            throw std::runtime_error(
                "Direct input format unsupported: color=" +
                std::to_string(unsigned(color_description.Format)) +
                " motion=" + std::to_string(unsigned(motion_description.Format)));
        phoenix::direct::Resources resources(color, motion);
        FrameEvidence proof;
        proof.queue = queue;
        try {
            proof.binding_lease = observation::resolve(cmd, &resources, proof, false);
        } catch (const std::exception& feature_entry) {
            throw std::runtime_error(std::string(feature_entry.what()) + " " +
                                     observation::command_refusal_detail(cmd));
        }
        check(observation::current_provider_resource(cmd, color) &&
                  observation::current_provider_resource(cmd, motion),
              "Direct provider input lacks fresh state or is explicitly deactivated");
        ID3D12Resource* worker_depth = nullptr;
        D3D12_RESOURCE_STATES worker_depth_state = D3D12_RESOURCE_STATE_COMMON;
        if ((own_route && addon_state.panel.execution_device != "native") ||
            (resolution.scaled() && addon_state.panel.upscaler != 0)) {
            worker_depth = parameter_resource(params, NVSDK_NGX_Parameter_Depth);
            if (!worker_depth)
                worker_depth = parameter_resource(params, "DLSSNR.Depth");
            if (worker_depth) {
                addon_state.bridge_inputs["worker_depth_candidate"] =
                    describe(worker_depth->GetDesc());
                const bool supported = phoenix::GuideAdapter::supported(worker_depth->GetDesc());
                const bool planar =
                    phoenix::GuideAdapter::copy_format(worker_depth->GetDesc().Format) ==
                    DXGI_FORMAT_R32G8X24_TYPELESS;
                const bool known =
                    planar ? observation::depth_plane_state(cmd, worker_depth, worker_depth_state)
                           : observation::epoch_state(cmd, worker_depth, worker_depth_state);
                const bool active =
                    planar ? known : observation::current_provider_resource(cmd, worker_depth);
                addon_state.bridge_inputs["worker_depth_reason"] = !supported ? "unsupported_format"
                                                                   : !known   ? "state_not_observed"
                                                                   : !active
                                                                       ? "resource_not_current"
                                                                       : "accepted";
                if (!supported || !known || !active)
                    worker_depth = nullptr;
            } else
                addon_state.bridge_inputs["worker_depth_reason"] = "parameter_missing";
            addon_state.bridge_inputs["worker_depth_observed"] = worker_depth != nullptr;
            if (worker_depth)
                addon_state.bridge_inputs["depth"] = describe(worker_depth->GetDesc());
        }
        if (addon_state.direct_runner && !addon_state.direct_runner->ready()) {
            ++addon_state.direct_busy;
            addon_state.live.reset_requested = true;
            addon_state.direct_status =
                "Waiting for previous Phoenix DLSS5 frame fence; game output preserved";
            return;
        }
        const bool nvidia_source_requested = addon_state.panel.mode == phoenix::nvidia ||
            (addon_state.panel.mode == phoenix::split &&
             (addon_state.panel.source_a == 1 || addon_state.panel.source_b == 1));
        if (!nvidia_source_requested && (addon_state.selfdrive.handle || addon_state.selfdrive.output ||
                                        !addon_state.selfdrive.refinement.empty())) {
            const bool idle = !addon_state.direct_runner || addon_state.direct_runner->idle();
            if (!phoenix_model_tools::retire_optional_feature(addon_state.selfdrive, idle,
                    [&](auto* handle) {
                        check(addon_state.selfdrive.release_fn != nullptr, "NVIDIA release entry unavailable");
                        check(NVSDK_NGX_SUCCEED(addon_state.selfdrive.release_fn(handle)), "NVIDIA feature retirement failed");
                    })) {
                addon_state.direct_status = "Waiting for the last NVIDIA frame before releasing its resources";
                return;
            }
            ++addon_state.nvidia_retirements;
            addon_state.direct_nr_ready = false;
            addon_state.direct_nr_status = "NVIDIA resources released; Phoenix is the displayed source";
        }
        const phoenix_model_tools::ModelSelectionState active_selection{
            addon_state.active_model_source, addon_state.active_model_folder,
            addon_state.active_model_dll, addon_state.active_model_id,
            addon_state.active_nvidia_dll_folder};
        const bool model_changed = own_route &&
            phoenix_model_tools::ModelSelectionState::read(addon_state.panel) != active_selection;
        const bool replace_runner = !addon_state.direct_runner || model_changed ||
            !addon_state.direct_runner->matches(queue, color_description, motion_description) ||
            !addon_state.direct_runner->architecture_matches(addon_state.panel) ||
            !addon_state.direct_runner->depth_matches(worker_depth);
        if (replace_runner) {
            // The old generation is never destroyed with borrowed GPU work in
            // flight. A candidate may allocate resources, but cannot record or
            // submit a frame before it is published below.
            if (addon_state.direct_runner && !addon_state.direct_runner->idle()) {
                ++addon_state.direct_busy;
                addon_state.direct_status = "Waiting for the current frame before changing model/settings";
                return;
            }
            initializing = true;
            std::unique_ptr<phoenix_bundle::Bundle> candidate_bundle;
            std::vector<phoenix_kernel::Binary> candidate_kernels;
            std::unique_ptr<phoenix::GpuTiming> candidate_timing;
            auto selected_source = addon_state.weight_source;
            try {
                if (own_route && (model_changed || !addon_state.bundle))
                    selected_source = phoenix_model_tools::selection(addon_state.panel, addon_state.configured_weight_source);
                if (native_network && (model_changed || !addon_state.bundle)) {
                    candidate_bundle = std::make_unique<phoenix_bundle::Bundle>(
                        addon_state.bundle_path,
                        [](const auto& path) { return sha_file(path.c_str()); }, selected_source, true);
                    check(candidate_bundle->weights.at("unknown_gap_bytes") == 0, "Direct weights incomplete");
                    candidate_kernels = phoenix_kernel::load_catalog(phoenix_native::catalog(*candidate_bundle));
                }
                auto* selected_bundle = candidate_bundle ? candidate_bundle.get() : addon_state.bundle.get();
                const auto& selected_kernels = candidate_bundle ? candidate_kernels : addon_state.candidates;
                const bool replaced = phoenix_model_tools::replace_idle_generation(
                    addon_state.direct_runner, [&] {
                        Json portable_plan = {{"width", resolution.width}, {"height", resolution.height}};
                        const Json* selected = &portable_plan;
                        if (native_network) {
                            check(selected_bundle != nullptr, "Native model bundle unavailable");
                            selected = &selected_bundle->select_frame(resolution.width, resolution.height,
                                [&](const Json& frame) { phoenix_native::preflight(frame, selected_kernels); });
                        }
                        auto choice = resolve_allocation_device(queue, color);
                        check(choice.resolved, "Direct allocation device unresolved");
                        std::unique_ptr<phoenix::NeuralExecutor> executor;
                        if (own_route && addon_state.panel.execution_device != "native") {
                            check(std::any_of(addon_state.execution_devices.begin(),
                                              addon_state.execution_devices.end(),
                                              [&](const auto& device_info) {
                                                  return device_info.id == addon_state.panel.execution_device;
                                              }),
                                  "Selected processing device is unavailable");
                            executor = std::make_unique<phoenix::portable::OffloadExecutor>(
                                choice.device.Get(), resolution.width, resolution.height,
                                addon_state.worker_python, addon_state.worker_script,
                                addon_state.worker_weights,
                                std::filesystem::u8path(addon_state.panel.execution_device).wstring(),
                                addon_state.output_directory, unsigned(addon_state.panel.offload_age_frames),
                                uint64_t(addon_state.panel.offload_age_ms), addon_state.worker_vulkan_build,
                                std::wstring(addon_state.panel.worker_engine.begin(),
                                             addon_state.panel.worker_engine.end()),
                                addon_state.worker_hip_build, addon_state.worker_sycl_build,
                                selected_source, unsigned(addon_state.panel.cpu_threads));
                        }
                        const auto packed_weights = native_network ? selected_bundle->load_weight_view()
                                                                  : phoenix_model::WeightView{};
                        auto replacement = std::make_unique<phoenix::direct::Runner>(
                            choice.device.Get(), queue, *selected, packed_weights, selected_kernels,
                            color_description.Format, motion_description.Format, addon_state.direct_srgb,
                            unsigned(addon_state.panel.refinement_passes), addon_state.panel.block_repeats,
                            std::move(executor), &addon_state.panel,
                            addon_state.owned_allocations.begin(unsigned(color_description.Width),
                                                                color_description.Height),
                            true, unsigned(color_description.Width), color_description.Height, std::filesystem::path{},
                            motion_description.Width != color_description.Width || motion_description.Height != color_description.Height,
                            [&](unsigned w,unsigned h){
                                check(selected_bundle!=nullptr,"Native chain model bundle unavailable");
                                return Json(selected_bundle->select_frame(w,h,[&](const Json& frame){phoenix_native::preflight(frame,selected_kernels);}));
                            });
                        candidate_timing = std::make_unique<phoenix::GpuTiming>(
                            choice.device.Get(), queue, addon_state.timing_archive);
                        return replacement;
                    }, [&](const auto& previous) {
                        addon_state.candidate_launches += previous.launches();
                        addon_state.completed_retired += previous.completed();
                    });
                check(replaced, "Executor replacement awaits its retirement fence");
                if (candidate_bundle) {
                    addon_state.bundle = std::move(candidate_bundle);
                    addon_state.candidates = std::move(candidate_kernels);
                }
                if (own_route) {
                    addon_state.weight_source = std::move(selected_source);
                    addon_state.active_model_source = addon_state.panel.model_source;
                    addon_state.active_model_folder = addon_state.panel.model_folder;
                    addon_state.active_model_dll = addon_state.panel.model_dll;
                    addon_state.active_model_id = addon_state.panel.model_id;
                    addon_state.active_nvidia_dll_folder = addon_state.panel.nvidia_dll_folder;
                }
                addon_state.model_switch_error.clear();
            } catch (const std::exception& failure) {
                if (model_changed && addon_state.direct_runner) {
                    // Keep the working executor and its authenticated bundle.
                    // Reject this request once; do not rehash/reload a bad DLL
                    // on every frame or label the old output as the new model.
                    active_selection.restore(addon_state.panel);
                    addon_state.model_switch_error = std::string("Model change refused; kept the loaded model: ") + failure.what();
                    initializing = false;
                    throw std::runtime_error(addon_state.model_switch_error);
                }
                throw;
            }
            addon_state.live.reset_requested = true;
            addon_state.depth_budget.reset_samples();
            addon_state.timing_settings_dirty = true;
            initializing = false;
            if (addon_state.live_timing) {
                addon_state.live_timing->budget(nullptr);
                addon_state.retired_timing.push_back(std::move(addon_state.live_timing));
            }
            addon_state.live_timing = std::move(candidate_timing);
            addon_state.live_timing->budget(&addon_state.depth_budget);
            phoenix_native::group_marker = [](ID3D12GraphicsCommandList* list, unsigned group) {
                if (auto& timing = state().live_timing) timing->group(list, group);
            };
            ++addon_state.generations;
            reshade::log::message(reshade::log::level::info,
                ("[Phoenix DLSS5] Executor generation " + std::to_string(addon_state.generations) +
                 ": " + (own_route ? addon_state.weight_source.model_id : std::string("NVIDIA")) +
                 " (previous generation retired)").c_str());
        }
        Controls controls;
        unsigned host_flags = 0;
        if (feature)
            if (const auto* observed = addon_state.ngx_observed.find(native_feature_key(feature)))
                host_flags = observed->create_flags;
        {
            unsigned flags = 0;
            if (NVSDK_NGX_SUCCEED(params->Get("DLSS.Feature.Create.Flags", &flags)))
                host_flags = flags;
        }
        controls.inverted = (host_flags & 8u) ? 1 : 0;
        auto get = [&](const char* key, auto& value) {
            auto saved = value;
            if (!NVSDK_NGX_SUCCEED(params->Get(key, &value)))
                value = saved;
        };
        get(NVSDK_NGX_Parameter_Reset, controls.reset);
        get(NVSDK_NGX_Parameter_MV_Scale_X, controls.motion_x);
        get(NVSDK_NGX_Parameter_MV_Scale_Y, controls.motion_y);
        {
            float hjx = 0.0f, hjy = 0.0f;
            get("Jitter.Offset.X", hjx);
            get("Jitter.Offset.Y", hjy);
            if (addon_state.direct_runner)
                addon_state.direct_runner->set_host_jitter(hjx, hjy);
        }
        {
            static bool host_probed = false;
            if (!host_probed) {
                host_probed = true;
                float jitter_x = 0, jitter_y = 0, normalized_jitter_x = 0, normalized_jitter_y = 0;
                const bool has_j = NVSDK_NGX_SUCCEED(params->Get("Jitter.Offset.X", &jitter_x)) &&
                                   NVSDK_NGX_SUCCEED(params->Get("Jitter.Offset.Y", &jitter_y));
                const bool has_nj =
                    NVSDK_NGX_SUCCEED(
                        params->Get("DLSSNR.Jitter.Offset.X", &normalized_jitter_x)) &&
                    NVSDK_NGX_SUCCEED(params->Get("DLSSNR.Jitter.Offset.Y", &normalized_jitter_y));
                char msg[256]{};
                snprintf(
                    msg, sizeof(msg),
                    "[Phoenix DLSS5] host probe: mvscale=%.4f,%.4f jitter=%d(%.4f,%.4f) nrjitter=%d(%.4f,%.4f) reset=%d",
                    double(controls.motion_x), double(controls.motion_y), has_j ? 1 : 0,
                    double(jitter_x), double(jitter_y), has_nj ? 1 : 0, double(normalized_jitter_x),
                    double(normalized_jitter_y), controls.reset);
                reshade::log::message(reshade::log::level::info, msg);
            }
        }
        unsigned developer_contract=0;
        params->Get("Phoenix.Appearance.Contract",&developer_contract);
        if(developer_contract==1) {
            get("DLSSNR.Intensity",controls.intensity);get("DLSSNR.Style",controls.style);
            get("DLSSNR.LocalToneStrength",controls.tone);get("DLSSNR.LocalStructureStrength",controls.structure);
            get("DLSSNR.SkinStructureStrength",controls.skin);get("DLSSNR.UseAutoMask",controls.auto_mask);
        }
        controls = addon_state.live.apply(controls);
        if (addon_state.direct_runner) {
            addon_state.direct_runner->poll_exposure();
            addon_state.direct_runner->poll_captures();
            phoenix::direct::CaptureFrame cap;
            while (addon_state.direct_runner->take_capture(cap))
                write_capture_frame(cap);
        }
        if (addon_state.last_mode != addon_state.panel.mode)
            addon_state.reset_hold = addon_state.panel.reset_after_switch;
        if (addon_state.last_mode != addon_state.panel.mode || addon_state.panel.force_reset ||
            addon_state.reset_hold > 0 ||
            (addon_state.panel.reset_every &&
             addon_state.recorded % unsigned(addon_state.panel.reset_every) == 0))
            controls.reset = 1;
        addon_state.depth_budget.configure(
            addon_state.panel.mode == phoenix::ours && addon_state.panel.experimental &&
                addon_state.panel.automatic_budget,
            unsigned(addon_state.panel.transformer_blocks),
            unsigned(addon_state.panel.minimum_blocks), addon_state.panel.budget_target_ms(),
            addon_state.panel.budget_mode == 1, addon_state.panel.block_repeats);
        unsigned blocks = addon_state.panel.experimental && addon_state.panel.mode == phoenix::ours
                              ? (addon_state.panel.automatic_budget
                                     ? addon_state.depth_budget.depth()
                                     : unsigned(addon_state.panel.transformer_blocks))
                              : 8u;
        if (blocks != addon_state.last_transformer_blocks)
            controls.reset = 1;
        if (addon_state.direct_feeder)
            addon_state.feeder_applied_reset = controls.reset != 0;
        if (addon_state.timing_settings_dirty) {
            Json key = Json::object();
            phoenix::visit_settings(addon_state.panel,
                                    [&](const char* name, auto& value) { key[name] = value; });
            for (const char* name : {"Backend", "Divider", "ScreenshotRequest", "CaptureBurst",
                                     "ExternalReload", "LogLevel"})
                key.erase(name);
            if (!own_route)
                key["ExecutionDevice"] = "native";
            if (own_route) {
                key["ModelId"] = addon_state.weight_source.model_id;
                key["LoadedModelSource"] = addon_state.weight_source.configuration();
            }
            key["InferenceWidth"] = resolution.width;
            key["InferenceHeight"] = resolution.height;
            addon_state.live_timing->settings(key.dump());
            addon_state.timing_settings_dirty = false;
        }
        phoenix::direct::network_timing_begin = [] {
            auto& timing = state().live_timing;
            if (timing)
                timing->own_begin();
        };
        phoenix::direct::network_timing_end = [] {
            auto& timing = state().live_timing;
            if (timing)
                timing->own_end();
        };
        addon_state.live_timing->begin(cmd, unsigned(addon_state.panel.mode), blocks,
                                       unsigned(color_description.Width),
                                       unsigned(color_description.Height));
        {
            float host_white = 100, host_exposure = 1;
            get("DLSSNR.DiffuseWhiteNits", host_white);
            get("DLSS.Pre.Exposure", host_exposure);
            addon_state.direct_runner->processing_frame(host_white, host_exposure,
                phoenix_present_stage ? phoenix_present_encoding :
                addon_state.direct_feeder && addon_state.direct_srgb ? 2 : 1);
        }
        // Self-hosted native NR, evaluated before our network so wipe sides
        // can stage its image. NR failure only withholds the NVIDIA source;
        // the DLSS5 Phoenix frame below still records.
        const int desired_nr_hdr = addon_state.panel.processing.enabled ? 0 : addon_state.panel.host_hdr;
        const bool need_nr = addon_state.panel.mode == phoenix::nvidia ||
                             (addon_state.panel.mode == phoenix::split &&
                              (addon_state.panel.source_a == 1 || addon_state.panel.source_b == 1));
        ID3D12Resource* nr_image = nullptr;
        // Creation records GPU work too. Keep NVIDIA's optional runtime idle
        // until a displayed source requests it, rather than initializing it
        // during startup resizes while Phoenix is the selected backend.
        if (need_nr && (!GetModuleHandleW(L"renodx-dlss5.addon64") && !GetModuleHandleW(L"renodx-dlss.addon64"))) {
            auto& selfdrive_state = addon_state.selfdrive;
            const auto now = GetTickCount64();
            const bool geom_changed =
                selfdrive_state.handle &&
                (selfdrive_state.in_w != resolution.width ||
                 selfdrive_state.in_h != resolution.height ||
                 selfdrive_state.bridge != addon_state.panel.host_bridge ||
                 selfdrive_state.hdr_mode != desired_nr_hdr ||
                 selfdrive_state.auto_exp != (addon_state.panel.nr_auto_exposure ? 1 : 0) ||
                 selfdrive_state.feature_id != unsigned(addon_state.panel.nr_feature) ||
                 selfdrive_state.preset != phoenix::nvidia_preset(addon_state.panel.nvidia_model,
                     addon_state.panel.model_id.empty() ? addon_state.weight_source.model().id : addon_state.panel.model_id) ||
                 selfdrive_state.passes != unsigned(addon_state.panel.refinement_passes) ||
                 selfdrive_state.source_choice != addon_state.panel.model_source + "\n" +
                     addon_state.panel.model_folder + "\n" + addon_state.panel.model_dll + "\n" + addon_state.panel.nvidia_dll_folder);
            try {
                if (addon_state.panel.uplift && (!selfdrive_state.handle || geom_changed) &&
                    (selfdrive_state.requested_feature != addon_state.panel.nr_feature ||
                     selfdrive_state.attempts == 0 ||
                     ngx_shim::selfdrive_retry_due(now, selfdrive_state.retry_after_tick))) {
                    const auto nr_owner = resolve_allocation_device(queue, color);
                    check(nr_owner.resolved, "phoenix NR allocation device unresolved");
                    selfdrive_direct_ensure(cmd, nr_owner.device.Get(), resolution.width,
                                            resolution.height, addon_state.panel.host_bridge,
                                            desired_nr_hdr,
                                            addon_state.panel.nr_auto_exposure,
                                            addon_state.panel.nr_feature, host_flags);
                }
                if (need_nr && addon_state.panel.uplift && selfdrive_state.handle &&
                    selfdrive_state.warmup == 0) {
                    addon_state.direct_runner->set_host_bridge(addon_state.panel.host_bridge != 0,
                                                               addon_state.panel.host_white_nits /
                                                                   100.0f);
                    {
                        addon_state.direct_runner->encode_shared_proxy(
                            cmd, color, proof.color_state, controls, addon_state.panel);
                        if (addon_state.panel.mode == phoenix::nvidia &&
                            addon_state.panel.log_level == 2) {
                            reshade::log::message(
                                reshade::log::level::info,
                                ("[Phoenix DLSS5] nr-frame: proxy encoded color=" +
                                 phoenix_describe(color, proof.color_state))
                                    .c_str());
                            phoenix_crumb("nr-frame: proxy encoded");
                        }
                        if (proof.restore_bindings)
                            proof.restore_bindings(proof.restore_context, cmd);
                    }
                    {
                        const float sd_tone = addon_state.panel.processing.enabled ? controls.tone : phoenix::host_strength(addon_state.live.enabled,
                                                                     addon_state.live.tone, 1.0f);
                        const float sd_structure = addon_state.panel.processing.enabled ? controls.structure : phoenix::host_strength(
                            addon_state.live.enabled, addon_state.live.structure, 1.0f);
                        float sd_skin = addon_state.panel.processing.enabled ? controls.skin : phoenix::host_strength(addon_state.live.advanced,
                                                                     addon_state.live.skin, -1.0f);
                        const unsigned sd_mask = addon_state.panel.processing.enabled
                            ? unsigned(controls.auto_mask != 0)
                            : ((!addon_state.live.advanced || addon_state.live.auto_mask) ? 1u : 0u);
                        if (!sd_mask) sd_skin = 0;
                        const unsigned sd_style = addon_state.panel.processing.enabled && !addon_state.panel.native_options
                            ? controls.style : unsigned(phoenix::host_strength(
                                addon_state.panel.native_options, addon_state.panel.style, 0));
                        unsigned host_ui_correction = 0;
                        get("DLSSNR.UICorrection", host_ui_correction);
                        const unsigned sd_uicorr = addon_state.panel.processing.enabled
                            ? unsigned(addon_state.panel.processing.correction(host_ui_correction != 0))
                            : unsigned(phoenix::host_strength(addon_state.panel.native_options, addon_state.panel.ui_correction, 1));
                        float sd_jx = 0.0f, sd_jy = 0.0f;
                        if (addon_state.panel.host_jitter) {
                            if (!NVSDK_NGX_SUCCEED(params->Get("Jitter.Offset.X", &sd_jx)))
                                sd_jx = 0.0f;
                            if (!NVSDK_NGX_SUCCEED(params->Get("Jitter.Offset.Y", &sd_jy)))
                                sd_jy = 0.0f;
                            if (sd_jx == 0.0f && sd_jy == 0.0f) {
                                float jitter_x = 0.0f, jitter_y = 0.0f;
                                if (NVSDK_NGX_SUCCEED(
                                        params->Get("DLSSNR.Jitter.Offset.X", &jitter_x)))
                                    sd_jx = jitter_x;
                                if (NVSDK_NGX_SUCCEED(
                                        params->Get("DLSSNR.Jitter.Offset.Y", &jitter_y)))
                                    sd_jy = jitter_y;
                            }
                        }
                        const unsigned sd_flags =
                            addon_state.panel.host_bridge ? host_flags | 0x41u : host_flags;
                        ID3D12Resource* sd_depth =
                            parameter_resource(params, NVSDK_NGX_Parameter_Depth);
                        if (!sd_depth)
                            sd_depth = parameter_resource(params, "DLSSNR.Depth");
                        if (!sd_depth &&
                            NVSDK_NGX_SUCCEED(params->Get(NVSDK_NGX_Parameter_Depth, &sd_depth)) &&
                            sd_depth)
                            ++selfdrive_state.depth_typed;
                        if (sd_depth)
                            ++selfdrive_state.depth_present;
                        D3D12_RESOURCE_STATES sd_depth_state =
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                        bool sd_depth_observed = false;
                        if (sd_depth) {
                            D3D12_RESOURCE_STATES known{};
                            if (observation::epoch_state(cmd, sd_depth, known)) {
                                sd_depth_state = known;
                                sd_depth_observed = true;
                                ++selfdrive_state.depth_proven;
                            }
                        }
                        auto* native_motion = addon_state.direct_runner->prepare_native_motion(
                            cmd, motion, proof.motion_state);
                        auto* native_depth = sd_depth_observed ? sd_depth : nullptr;
                        if (resolution.scaled() && native_depth)
                            native_depth = addon_state.direct_runner->prepare_native_depth(
                                cmd, native_depth, sd_depth_state);
                        if (proof.restore_bindings)
                            proof.restore_bindings(proof.restore_context, cmd);
                        addon_state.live_timing->native_begin();
                        selfdrive_direct_run(
                            cmd, addon_state.direct_runner->proxy_image(), native_motion,
                            resolution.scaled() ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                                : proof.motion_state,
                            controls.reset, controls.motion_x, controls.motion_y,
                            addon_state.panel.processing.enabled && !addon_state.panel.native_options
                                ? controls.intensity : addon_state.panel.intensity,
                            sd_tone, sd_structure, sd_skin, sd_mask,
                            sd_style, sd_uicorr, addon_state.panel.host_globaltone,
                            addon_state.panel.processing.enabled ? 1.f : addon_state.panel.host_exposure,
                            addon_state.panel.processing.enabled ? 1.f : addon_state.panel.host_exposure_scale,
                            sd_jx * resolution.motion_x(), sd_jy * resolution.motion_y(), sd_flags,
                            native_depth,
                            resolution.scaled() ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                                : sd_depth_state,
                            controls.inverted, selfdrive_state.in_w, selfdrive_state.in_h);
                        nr_image = selfdrive_state.result();
                        addon_state.live_timing->native_end();
                        if (addon_state.panel.mode == phoenix::nvidia &&
                            addon_state.panel.log_level == 2) {
                            reshade::log::message(
                                reshade::log::level::info,
                                ("[Phoenix DLSS5] nr-frame: NR evaluated output=" +
                                 phoenix_describe(nr_image, D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
                                    .c_str());
                            phoenix_crumb("nr-frame: NR evaluated");
                        }
                    }
                    if (nr_image)
                        addon_state.direct_nr_status.clear();
                    else if (addon_state.panel.mode == phoenix::nvidia)
                        addon_state.direct_nr_status = "NR resolve staging holds the NVIDIA DLSS5 source";
                } else if (need_nr && !addon_state.panel.uplift)
                    addon_state.direct_nr_status =
                        "Neural rendering off (Uplift disabled); NVIDIA DLSS5 sources withheld";
                else if (need_nr && selfdrive_state.handle)
                    addon_state.direct_nr_status = "NR feature warming up";
            } catch (const std::exception& feature_entry) {
                addon_state.direct_nr_status = feature_entry.what();
                if (++selfdrive_state.abandons > 12 && !selfdrive_state.failed) {
                    selfdrive_state.failed = true;
                    reshade::log::message(
                        reshade::log::level::warning,
                        "[Phoenix DLSS5] self-drive: NR abandoned after repeated refusals");
                } else if (selfdrive_state.abandons <= 3)
                    reshade::log::message(reshade::log::level::info,
                                          ("[Phoenix DLSS5] self-drive deferred: " +
                                           std::string(feature_entry.what()))
                                              .c_str());
                selfdrive_state.retry_after_tick = now + ngx_shim::kSelfDriveRetryCooldownMs;
            } catch (...) {
                selfdrive_state.failed = true;
                addon_state.direct_nr_status = "NR fault";
            }
        }
        addon_state.direct_nr_ready =
            addon_state.selfdrive.handle != nullptr && !addon_state.selfdrive.failed;
        // Warmup drains on claimed frames, not recorded ones: warmup frames
        // themselves defer, so gating the drain on recording livelocks.
        if (addon_state.selfdrive.warmup > 0)
            --addon_state.selfdrive.warmup;
        // Gate on fresh readiness: split frames that request NVIDIA sources
        // must first get their chance to create the feature above.
        check(phoenix::direct::comparison_supported(addon_state.panel, nr_image != nullptr),
              addon_state.direct_nr_status.empty()
                  ? "Independent comparison needs a live NR feature for NVIDIA DLSS5 sources"
                  : addon_state.direct_nr_status.c_str());
        if (addon_state.panel.mode == phoenix::nvidia) {
            // Fullscreen native NR: our network stays idle (history resumes
            // with a mandatory reset on the way back). The NR answer lives
            // in proxy space, so it is resolved onto the engine linear
            // output, never copied raw.
            check(nr_image, addon_state.direct_nr_status.empty()
                                ? "NR feature unavailable"
                                : addon_state.direct_nr_status.c_str());
            {
                char entry[64];
                snprintf(entry, sizeof(entry), "nr-frame %llu: claim entry",
                         addon_state.routed_frames);
                phoenix_crumb(entry);
            }
            // Same resource contract as the recorded path: refuse exotic
            // layouts here instead of submitting them to the driver.
            for (auto* resource : {color, motion, nr_image}) {
                const auto resource_description = resource->GetDesc();
                check(resource_description.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
                          resource_description.MipLevels == 1 &&
                          resource_description.DepthOrArraySize == 1 &&
                          resource_description.SampleDesc.Count == 1,
                      "Direct NVIDIA DLSS5 resource layout unsupported");
            }
            check(color != motion && color != nr_image, "Direct NVIDIA DLSS5 aliased inputs");
            {
                char timing_text[128];
                snprintf(timing_text, sizeof(timing_text), "nr-frame: list type=%d queue type=%d",
                         int(cmd->GetType()), int(queue->GetDesc().Type));
                if (addon_state.mode_frames[size_t(addon_state.panel.mode)] == 0)
                    reshade::log::message(reshade::log::level::info,
                                          ("[Phoenix DLSS5] " + std::string(timing_text)).c_str());
                phoenix_crumb(timing_text);
            }
            check(cmd->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT &&
                      queue->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT,
                  "Direct NVIDIA DLSS5 needs a direct list and queue");
            if (addon_state.mode_frames[size_t(addon_state.panel.mode)] == 0)
                reshade::log::message(
                    reshade::log::level::info,
                    ("[Phoenix DLSS5] nr-frame: first resolve color=" +
                     phoenix_describe(color, proof.color_state) + " proxy=" +
                     phoenix_describe(addon_state.direct_runner->proxy_image(),
                                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) +
                     " nr=" + phoenix_describe(nr_image, D3D12_RESOURCE_STATE_UNORDERED_ACCESS) +
                     " motion=" + phoenix_describe(motion, proof.motion_state))
                        .c_str());
            recording = true;
            addon_state.direct_runner->prepare_upscale(cmd, controls, worker_depth,
                                                       worker_depth_state);
            addon_state.direct_runner->note_claimed(cmd, color, motion);
            {
                /* timing opens with the NR section above */
                addon_state.direct_runner->resolve_fullscreen(
                    cmd, color, proof.color_state, addon_state.direct_runner->proxy_image(),
                    nr_image);
                if (addon_state.panel.log_level == 2)
                    reshade::log::message(reshade::log::level::info,
                                          "[Phoenix DLSS5] nr-frame: resolve recorded");
                phoenix_crumb("nr-frame: resolve recorded");
            }
            addon_state.live_timing->upscaler_status(addon_state.direct_runner->upscale_status());
            addon_state.live_timing->end();
            if (proof.restore_bindings)
                proof.restore_bindings(proof.restore_context, cmd);
            phoenix_crumb("nr-frame: bindings restored");
            if (addon_state.last_mode != addon_state.panel.mode) {
                ++addon_state.switches;
                record_event("direct_mode", {{"mode", addon_state.panel.mode},
                                             {"frame", addon_state.routed_frames}});
            }
            addon_state.last_mode = addon_state.panel.mode;
            ++addon_state.recorded;
            ++addon_state.routed_frames;
            ++addon_state.mode_frames[size_t(addon_state.panel.mode)];
            if (addon_state.reset_hold > 0)
                --addon_state.reset_hold;
            if (addon_state.selfdrive.warmup > 0)
                --addon_state.selfdrive.warmup;
            addon_state.live.recorded();
            addon_state.direct_status =
                "Native NR output (resolved) written back to the game DLSS output";
            addon_state.displayed = addon_state.panel;
            addon_state.displayed.mode = phoenix::nvidia;
            addon_state.display_tick = GetTickCount64();
            return;
        }
        addon_state.direct_runner->set_nr(nr_image);
        auto developer_textures=harvest_appearance(cmd,params,proof);
        if(addon_state.panel.processing.enabled) {
            unsigned host_ui=0;params->Get("DLSSNR.UICorrection",&host_ui);
            controls.ui_correction = addon_state.panel.processing.correction(host_ui!=0);
            if(controls.ui_correction && !developer_textures.ui && !developer_textures.alpha) {
                check(addon_state.panel.processing.ui_policy != 2,
                      "UI correction On requires an explicit UI or alpha coverage plane");
                controls.ui_correction=0;
            }
            if(!controls.auto_mask)controls.skin=0;
        }
        unsigned appearance_generation=0;params->Get("Phoenix.Appearance.Generation",&appearance_generation);
        const auto before = addon_state.direct_runner->launches();
        recording = true;
        addon_state.direct_runner->record(cmd, color, motion, proof, controls, addon_state.panel,
                                          blocks, worker_depth, worker_depth_state,&developer_textures,appearance_generation,
                                          addon_state.depth_budget.epoch());
        addon_state.live_timing->upscaler_status(addon_state.direct_runner->upscale_status());
        addon_state.live_timing->end();
        const auto launches = addon_state.direct_runner->launches() - before;
        if (addon_state.last_mode != addon_state.panel.mode) {
            ++addon_state.switches;
            record_event("direct_mode",
                         {{"mode", addon_state.panel.mode}, {"frame", addon_state.routed_frames}});
        }
        addon_state.last_mode = addon_state.panel.mode;
        addon_state.last_transformer_blocks = blocks;
        ++addon_state.recorded;
        ++addon_state.routed_frames;
        ++addon_state.mode_frames[size_t(addon_state.panel.mode)];
        if (addon_state.reset_hold > 0)
            --addon_state.reset_hold;
        addon_state.mode_own[size_t(addon_state.panel.mode)] += launches;
        ++addon_state.depth_frames[blocks];
        addon_state.live.recorded();
        addon_state.direct_status = "Own network output written back to the game DLSS output";
        if (addon_state.direct_runner->external()) {
            const auto execution = addon_state.direct_runner->execution_status();
            addon_state.worker_budget.sample(addon_state.depth_budget, execution,
                addon_state.panel.model_id.empty() ? addon_state.weight_source.model_id : addon_state.panel.model_id);
            const auto status = execution.value("state", std::string());
            std::string label = addon_state.panel.execution_device;
            for (const auto& device_info : addon_state.execution_devices)
                if (device_info.id == label) {
                    label = device_info.label;
                    break;
                }
            const auto key = Json{{"device", addon_state.panel.execution_device},
                                  {"engine", addon_state.panel.worker_engine},
                                  {"cpu_threads", addon_state.panel.cpu_threads},
                                  {"width", resolution.width},
                                  {"height", resolution.height},
                                  {"output_width", color_description.Width},
                                  {"output_height", color_description.Height},
                                  {"aspect", addon_state.panel.resolution_aspect},
                                  {"blocks", blocks},
                                  {"passes", addon_state.panel.refinement_passes},
                                  {"repeats", addon_state.panel.block_repeats}}
                                 .dump();
            addon_state.worker_timing.sample(key,
                                             label + " / " + addon_state.panel.worker_engine +
                                                 " / " + std::to_string(color_description.Width) +
                                                 "x" + std::to_string(color_description.Height),
                                             execution);
            if (status == "composing_reprojected_result" || status == "presenting_reprojected_result")
                addon_state.direct_status =
                    "Phoenix DLSS5 worker result reprojected onto the current frame";
            else if (status == "failed_input_preserved")
                addon_state.direct_status =
                    "Worker failed: " + execution.value("error", std::string());
            else if (status == "missing_linear_depth_guides")
                addon_state.direct_status =
                    "Waiting for supported, observed depth; game output preserved";
            else
                addon_state.direct_status =
                    "Waiting for a fresh Phoenix DLSS5 worker result; game output preserved";
        }
        addon_state.displayed = addon_state.panel;
        addon_state.displayed.mode = addon_state.panel.mode;
        addon_state.display_tick = GetTickCount64();
        if (addon_state.recorded == 1 || addon_state.recorded % 300 == 0)
            reshade::log::message(
                reshade::log::level::info,
                ("[Phoenix DLSS5] Direct recorded=" + std::to_string(addon_state.recorded) +
                 " own launches/frame=" + std::to_string(launches) +
                 " exposure flips=" + std::to_string(addon_state.direct_runner->exposure_flips()) +
                 " max=" + std::to_string(addon_state.direct_runner->exposure_max_delta_pct()) +
                 "%")
                    .c_str());
        if (addon_state.panel.mode == phoenix::split &&
            (addon_state.panel.source_a == 1 || addon_state.panel.source_b == 1) &&
            addon_state.recorded % 600 == 0 && addon_state.live_timing) {
            const auto timing = addon_state.live_timing->report();
            for (const auto& row : timing.at("rows")) {
                if (row.at("mode").get<unsigned>() != unsigned(phoenix::split))
                    continue;
                const auto &own = row.at("phoenix"), nat = row.at("nvidia");
                char timing_msg[224]{};
                snprintf(
                    timing_msg, sizeof(timing_msg),
                    "[Phoenix DLSS5] Split timings: phoenix=%.3f ms (%llu samples) nvidia=%.3f ms (%llu samples)",
                    own.at("mean_ms").get<double>(), own.at("samples").get<uint64_t>(),
                    nat.at("mean_ms").get<double>(), nat.at("samples").get<uint64_t>());
                reshade::log::message(reshade::log::level::info, timing_msg);
            }
        }
    } catch (const std::exception& feature_entry) {
        if (addon_state.live_timing)
            addon_state.live_timing->end();
        ++addon_state.direct_deferred;
        addon_state.direct_status = feature_entry.what();
        addon_state.live.reset_requested = true;
        if (!recording) {
            if(initializing)recover_initialization_failure(feature_entry.what());
            // Deferred frames leave the engine output untouched, so the
            // screen shows the game input: label it truthfully and keep the
            // heartbeat fresh instead of letting a stale wipe label linger.
            addon_state.displayed = addon_state.panel;
            addon_state.displayed.mode = phoenix::input;
            addon_state.display_tick = GetTickCount64();
        }
        if (recording) {
            if (addon_state.direct_runner)
                addon_state.direct_runner->invalidate();
            addon_state.terminal = true;
            error(feature_entry.what());
        }
        if (GetTickCount64() - addon_state.direct_status_tick > 3000) {
            addon_state.direct_status_tick = GetTickCount64();
            reshade::log::message(
                reshade::log::level::warning,
                ("[Phoenix DLSS5] Direct deferred: " + addon_state.direct_status).c_str());
        }
    }
}
void command_reset(ID3D12GraphicsCommandList* cmd, bool success) {
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    if (success && addon_state.present_guides) addon_state.present_guides->abandoned(cmd);
    if (success && addon_state.direct_runner && addon_state.direct_runner->pending(cmd)) {
        addon_state.direct_runner->invalidate();
        addon_state.terminal = true;
        error("Direct frame command reset before observed submission");
    }
    if (success) {
        addon_state.submitted_lists.erase(cmd);
        addon_state.states.reset(uint64_t(cmd));
        if (addon_state.live_timing)
            addon_state.live_timing->abandoned(cmd);
        for (const auto& timer : addon_state.retired_timing)
            timer->abandoned(cmd);
    }
    auto handle = [&](const auto& session) {
        if (!session || session->command() != cmd)
            return;
        try {
            if (success)
                session->abandoned(cmd);
            else
                session->uncertain_submission();
        } catch (...) {
            session->uncertain_submission();
        }
        addon_state.terminal = true;
        ++addon_state.resets;
    };
    handle(addon_state.endpoint.session);
    for (const auto& session : addon_state.retired)
        handle(session);
}
void command_close(ID3D12GraphicsCommandList* cmd, bool success) {
    if (success)
        return;
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    if (addon_state.endpoint.session && addon_state.endpoint.session->command() == cmd) {
        addon_state.endpoint.session->uncertain_submission();
        addon_state.terminal = true;
        error("phoenix candidate command Close failed");
    }
}
void destroy_present_runtime(reshade::api::effect_runtime* runtime) {
    auto& value=state();std::lock_guard<std::recursive_mutex> lock(value.mutex);
    if(value.present_runtime==runtime) {
        value.present_bridge.reset();value.present_runtime=nullptr;
        value.live.reset_requested=true;
    }
}
void init_resource(reshade::api::device* device, const reshade::api::resource_desc&,
                   const reshade::api::subresource_data*,
                   reshade::api::resource_usage initial_state, reshade::api::resource resource) {
    if (device->get_api() != reshade::api::device_api::d3d12 || !resource.handle)
        return;
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    try {
        addon_state.states.created(resource.handle, uint32_t(initial_state));
    } catch (...) {
    }
}
void destroy_resource(reshade::api::device* device, reshade::api::resource resource) {
    if (device->get_api() != reshade::api::device_api::d3d12 || !resource.handle)
        return;
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    addon_state.states.destroyed(resource.handle);
    observation::forget_resource(reinterpret_cast<ID3D12Resource*>(resource.handle));
}
void barrier(reshade::api::command_list* list, uint32_t count,
             const reshade::api::resource* resources, const reshade::api::resource_usage*,
             const reshade::api::resource_usage* new_states) {
    if (!list || list->get_device()->get_api() != reshade::api::device_api::d3d12)
        return;
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    const auto native = list->get_native();
    for (uint32_t iteration_index = 0; iteration_index < count; ++iteration_index) {
        if (!resources[iteration_index].handle)
            continue;
        try {
            addon_state.states.transition(native, resources[iteration_index].handle,
                                          uint32_t(new_states[iteration_index]));
        } catch (...) {
        }
    }
}
void init_command(reshade::api::command_list* list) {
    if (list->get_device()->get_api() != reshade::api::device_api::d3d12)
        return;
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    try {
        configure();
        if (!addon_state.automatic)
            return;
        auto* cmd = reinterpret_cast<ID3D12GraphicsCommandList*>(list->get_native());
        if (cmd->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT)
            observation::install(cmd, true);
    } catch (const std::exception& feature_entry) {
        addon_state.terminal = true;
        error(feature_entry.what());
    }
}
void destroy_command(reshade::api::command_list* list) {
    if (list->get_device()->get_api() != reshade::api::device_api::d3d12)
        return;
    auto* cmd = reinterpret_cast<ID3D12GraphicsCommandList*>(list->get_native());
    command_reset(cmd, true);
    observation::erase(cmd);
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    addon_state.states.destroyed_list(uint64_t(cmd));
    addon_state.ownership.erase(cmd);
    for (auto entry_iterator = addon_state.originals.begin();
         entry_iterator != addon_state.originals.end();)
        entry_iterator = (entry_iterator->first == cmd || entry_iterator->second == cmd)
                             ? addon_state.originals.erase(entry_iterator)
                             : std::next(entry_iterator);
}
void init_queue(reshade::api::command_queue* queue) {
    if (queue->get_device()->get_api() != reshade::api::device_api::d3d12)
        return;
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    addon_state.queues.push_back(queue);
    try {
        configure();
        if (!addon_state.automatic)
            return;
        auto* native = reinterpret_cast<ID3D12CommandQueue*>(queue->get_native());
        install_submission_observer(native);
        if (!addon_state.adapter) {
            ComPtr<ID3D12Device> queue_device;
            if (SUCCEEDED(native->GetDevice(IID_PPV_ARGS(&queue_device))) && queue_device) {
                const auto luid = queue_device->GetAdapterLuid();
                ComPtr<IDXGIFactory4> factory;
                if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) && factory) {
                    ComPtr<IDXGIAdapter1> found;
                    if (SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&found))) && found)
                        found.As(&addon_state.adapter);
                }
            }
        }
        if (native->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT)
            return;
        auto* immediate = queue->get_immediate_command_list();
        if (immediate)
            observation::install(
                reinterpret_cast<ID3D12GraphicsCommandList*>(immediate->get_native()));
        install_native(native);
    } catch (const std::exception& feature_entry) {
        addon_state.terminal = true;
        error(feature_entry.what());
    }
}
void destroy_queue(reshade::api::command_queue* queue) {
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    auto* native = reinterpret_cast<ID3D12CommandQueue*>(queue->get_native());
    for (auto cursor = addon_state.ownership.begin(); cursor != addon_state.ownership.end();)
        cursor = cursor->second.queue == native ? addon_state.ownership.erase(cursor)
                                                : std::next(cursor);
    addon_state.states.destroyed_queue(uint64_t(native));
    if (addon_state.endpoint.session &&
        uint64_t(addon_state.endpoint.session->queue()) == queue->get_native())
        retire();
    addon_state.queues.erase(
        std::remove(addon_state.queues.begin(), addon_state.queues.end(), queue),
        addon_state.queues.end());
}
// Presented-frame accounting for a measured run. The window opens after the
// requested warmup and closes on its own; when it closes the candidate budget
// is spent, so the engine simply keeps its own output for the rest of the run.
bool measure_window(uint64_t now) {
    auto& addon_state = state();
    if (!addon_state.measure)
        return true;
    ++addon_state.presents;
    if (!addon_state.first_present_tick)
        addon_state.first_present_tick = now;
    if (!addon_state.window_start_tick) {
        if (now - addon_state.first_present_tick < addon_state.warmup_ms)
            return false;
        addon_state.window_start_tick = now;
        addon_state.window_claimed_at_start = addon_state.claimed;
        addon_state.window_completed_at_start = addon_state.submitted;
        if (addon_state.timestamps)
            addon_state.timestamps->restart();
        return false;
    }
    if (addon_state.measurement_complete)
        return false;
    ++addon_state.window_presents;
    if (now - addon_state.window_start_tick >= addon_state.measure_ms) {
        addon_state.window_end_tick = now;
        addon_state.measurement_complete = true;
        addon_state.frame_limit =
            unsigned(addon_state.submitted); // spend the budget; native resumes
        return true;
    }
    return false;
}

void describe_swapchain(reshade::api::swapchain* swapchain, Json& detail) {
    auto& addon_state = state();
    detail["hwnd"] = uint64_t(uintptr_t(swapchain->get_hwnd()));
    detail["back_buffer_count"] = swapchain->get_back_buffer_count();
    const auto space = unsigned(swapchain->get_color_space());
    detail["color_space"] = space;
    detail["color_space_name"] = space == 1   ? "srgb"
                                 : space == 2 ? "scrgb"
                                 : space == 3 ? "hdr10_pq"
                                 : space == 4 ? "hdr10_hlg"
                                              : "unknown";
    addon_state.last_color_space = space;
    if (auto* device = swapchain->get_device()) {
        const auto back = swapchain->get_back_buffer(0);
        if (back.handle) {
            const auto desc = device->get_resource_desc(back);
            detail["width"] = desc.texture.width;
            detail["height"] = desc.texture.height;
            detail["format"] = unsigned(desc.texture.format);
            addon_state.last_backbuffer_width = desc.texture.width;
            addon_state.last_backbuffer_height = desc.texture.height;
            addon_state.last_backbuffer_format = unsigned(desc.texture.format);
        }
    }
}
void init_swapchain(reshade::api::swapchain* swapchain, bool resize) {
    if (!swapchain || swapchain->get_device()->get_api() != reshade::api::device_api::d3d12)
        return;
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    try {
        configure();
        if (!addon_state.automatic)
            return;
        ++addon_state.swapchains_created;
        if (resize)
            ++addon_state.swapchain_resizes;
        Json detail;
        detail["resize"] = resize;
        describe_swapchain(swapchain, detail);
        record_event("swapchain_init", std::move(detail));
    } catch (const std::exception& feature_entry) {
        addon_state.terminal = true;
        error(feature_entry.what());
    }
}
void destroy_swapchain(reshade::api::swapchain* swapchain, bool resize) {
    if (!swapchain || swapchain->get_device()->get_api() != reshade::api::device_api::d3d12)
        return;
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    try {
        if (!addon_state.automatic)
            return;
        ++addon_state.swapchains_destroyed;
        Json detail;
        detail["resize"] = resize;
        record_event("swapchain_destroy", std::move(detail));
    } catch (...) {
    }
}
bool set_fullscreen_state(reshade::api::swapchain* swapchain, bool fullscreen, void* monitor) {
    if (!swapchain || swapchain->get_device()->get_api() != reshade::api::device_api::d3d12)
        return false;
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    try {
        if (!addon_state.automatic)
            return false;
        ++addon_state.fullscreen_transitions;
        addon_state.last_fullscreen = fullscreen;
        Json detail;
        detail["fullscreen"] = fullscreen;
        detail["monitor"] = uint64_t(uintptr_t(monitor));
        record_event("fullscreen_state", std::move(detail));
    } catch (...) {
    }
    return false; // observation only; never change what the engine asked for
}

void present(reshade::api::command_queue* queue, reshade::api::swapchain* swapchain, const reshade::api::rect*,
             const reshade::api::rect*, uint32_t, const reshade::api::rect*) {
    if(queue && queue->get_device()->get_api()!=reshade::api::device_api::d3d12)return;
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    try {
        if (addon_state.runtime_recovery && addon_state.runtime_recovery->pending()) {
            // Bypass is a valid stable configuration. It cannot accumulate neural
            // completions, so prove its harmless presentation path separately.
            const bool bypass = addon_state.panel.mode == phoenix::input || !addon_state.panel.uplift;
            if (bypass && !addon_state.terminal && addon_state.live_loaded)
                addon_state.runtime_recovery->completed_frame();
            uint64_t completed = addon_state.completed_retired;
            if (addon_state.direct_runner) completed += addon_state.direct_runner->completed();
            if (addon_state.endpoint.session) completed += addon_state.endpoint.session->completed();
            if (completed > addon_state.recovery_completed) {
                const auto count = std::min<uint64_t>(30, completed - addon_state.recovery_completed);
                for (uint64_t i = 0; i < count; ++i) addon_state.runtime_recovery->completed_frame();
            }
            addon_state.recovery_completed = completed;
        }
        configure();
        if (!addon_state.automatic)
            return;
        if (!addon_state.terminal)
            install_native(nullptr);
        if (!addon_state.terminal && addon_state.direct_game && !addon_state.direct_feeder &&
            addon_state.panel.processing.enabled && addon_state.panel.processing.hook == 3 &&
            addon_state.panel.uplift && addon_state.present_runtime && swapchain && queue &&
            addon_state.present_runtime->get_hwnd() == swapchain->get_hwnd()) {
            struct PresentScope { PresentScope(){phoenix_present_stage=true;} ~PresentScope(){phoenix_present_stage=false;} } scope;
            try {
                auto* runtime = addon_state.present_runtime;
                check(runtime->get_command_queue() == queue, "Present requires the observed graphics queue");
                const auto space=swapchain->get_color_space();
                phoenix_present_encoding = space==reshade::api::color_space::hdr10_st2084 ? 3 :
                    space==reshade::api::color_space::extended_srgb_linear ? 4 : 2;
                const auto back = swapchain->get_current_back_buffer();
                const auto desc = queue->get_device()->get_resource_desc(back);
                if (addon_state.present_bridge && !addon_state.present_bridge->matches(desc))
                    addon_state.present_bridge.reset();
                if (!addon_state.present_bridge)
                    addon_state.present_bridge = std::make_unique<phoenix_compat::D3D12Bridge>(runtime,desc);
                addon_state.present_bridge->frame(back,addon_state.panel,true);
            } catch (const std::exception& e) {
                addon_state.direct_status = std::string("Present deferred: ")+e.what();
                addon_state.live.reset_requested=true;
            }
        }
        if (!addon_state.present_runtime || !swapchain ||
            addon_state.present_runtime->get_hwnd()==swapchain->get_hwnd()) ++addon_state.present_serial;
        collect();
        poll_route_pairs();
        const auto now = GetTickCount64();
        if (addon_state.panel.budget_mode == 1 && addon_state.panel.experimental &&
            addon_state.panel.automatic_budget) {
            if (addon_state.recorded != addon_state.budget_recorded) {
                addon_state.budget_recorded = addon_state.recorded;
                addon_state.budget_record_tick = now;
            }
            DWORD foreground_pid = 0;
            GetWindowThreadProcessId(GetForegroundWindow(), &foreground_pid);
            const bool budget_active =
                addon_state.panel.mode == phoenix::ours && addon_state.last_mode == phoenix::ours &&
                addon_state.panel.uplift && !addon_state.terminal &&
                foreground_pid == GetCurrentProcessId() && addon_state.budget_record_tick &&
                now - addon_state.budget_record_tick <= 250;
            const auto budget_now =
                uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now().time_since_epoch())
                             .count());
            addon_state.depth_budget.present(addon_state.last_transformer_blocks, budget_now,
                                             budget_active);
        } else {
            addon_state.budget_recorded = addon_state.recorded;
            addon_state.budget_record_tick = 0;
            addon_state.depth_budget.present(addon_state.last_transformer_blocks, 0, false);
        }
        // Count intervals after the initial timestamp, not the initial event.
        if (!addon_state.fps_tick) {
            addon_state.fps_tick = now;
            addon_state.fps_presents = 0;
        } else
            ++addon_state.fps_presents;
        if (now - addon_state.fps_tick >= 1000) {
            addon_state.present_fps =
                double(addon_state.fps_presents) * 1000.0 / double(now - addon_state.fps_tick);
            addon_state.fps_presents = 0;
            addon_state.fps_tick = now;
        }
        if (addon_state.live_timing)
            addon_state.live_timing->poll();
        for (const auto& timer : addon_state.retired_timing)
            timer->poll();
        addon_state.retired_timing.erase(
            std::remove_if(addon_state.retired_timing.begin(), addon_state.retired_timing.end(),
                           [](const auto& timer) { return timer->idle(); }),
            addon_state.retired_timing.end());
        if (!addon_state.last_memory_tick || now - addon_state.last_memory_tick >= 250) {
            addon_state.last_memory_tick = now;
            sample_memory();
        }
        sample_series(now); // owns its own interval, which widens as it decimates
        const bool closing = measure_window(now);
        if (addon_state.timestamps && addon_state.endpoint.session)
            addon_state.timestamps->poll(addon_state.endpoint.session->fence_completed());
        if (addon_state.measure || addon_state.interactive) {
            // Writing evidence every frame would be measured as our own cost.
            if ((addon_state.interactive || !closing) && addon_state.last_summary_tick &&
                now - addon_state.last_summary_tick < 500)
                return;
            addon_state.last_summary_tick = now;
        }
#if CNR_PHOENIX_DIAGNOSTIC
        if (!addon_state.measure)
            poll_diagnostic();
        CnrPHOENIXSummary(
            (addon_state.output_directory / "production_game_phoenix_diagnostic_summary.json")
                .c_str());
#else
        CnrPHOENIXSummary(
            (addon_state.output_directory / "production_game_phoenix_summary.json").c_str());
#endif
    } catch (const std::exception& feature_entry) {
        addon_state.terminal = true;
        error(feature_entry.what());
#if CNR_PHOENIX_DIAGNOSTIC
        if (addon_state.automatic)
            CnrPHOENIXSummary(
                (addon_state.output_directory / "production_game_phoenix_diagnostic_summary.json")
                    .c_str());
#endif
    }
}

template <class T>
void save_phoenix_value(reshade::api::effect_runtime* runtime, const char* section, const char* key,
                        const T& value) {
    if constexpr (std::is_same_v<T, std::string>)
        reshade::set_config_value(runtime, section, key, value.c_str());
    else
        reshade::set_config_value(runtime, section, key, value);
}
template <class T>
void load_phoenix_value(reshade::api::effect_runtime* runtime, const char* section, const char* key,
                        T& value) {
    if constexpr (std::is_same_v<T, std::string>) {
        char bytes[4096]{};
        size_t size = sizeof(bytes);
        if (reshade::get_config_value(runtime, section, key, bytes, &size))
            value.assign(bytes, strnlen_s(bytes, sizeof(bytes)));
    } else
        reshade::get_config_value(runtime, section, key, value);
}
Json panel_json(phoenix::Settings settings) {
    Json result = Json::object();
    phoenix::visit_settings(settings, [&](const char* key, auto& value) { result[key] = value; });
    return result;
}
bool restore_panel_json(const Json& stored, phoenix::Settings& settings) {
    auto candidate = settings;
    try {
        phoenix::visit_settings(candidate, [&](const char* key, auto& value) {
            if (stored.contains(key)) value = stored.at(key).template get<std::remove_reference_t<decltype(value)>>();
        });
    } catch (const Json::exception&) { return false; }
    if (!candidate.valid()) return false;
    settings = std::move(candidate); return true;
}
bool recover_initialization_failure(const char* text) noexcept {
    auto& value=state();
    if(!value.runtime_recovery||!value.panel.pipeline.recovery||!value.runtime_recovery->pending())return false;
    // Initialization has not recorded a borrowed game list. A previously
    // submitted executor may only be destroyed after its own fence retires.
    if(value.direct_runner&&!value.direct_runner->idle())return false;
    try{
        const auto fallback=value.runtime_recovery->failure(text);if(!fallback)return false;
        auto restored=value.panel;
        if(!restore_panel_json(*fallback,restored))return false;
        value.direct_runner.reset();value.bundle.reset();value.candidates.clear();
        value.panel=std::move(restored);value.live.reset_requested=true;value.timing_settings_dirty=true;
        value.recovery_notice="Provider initialization failed; restored last working settings before retrying.";
        if(value.present_runtime)phoenix::visit_settings(value.panel,[&](const char* key,auto& setting){save_phoenix_value(value.present_runtime,"Phoenix",key,setting);});
        reshade::log::message(reshade::log::level::warning,value.recovery_notice.c_str());return true;
    }catch(const std::exception& failure){reshade::log::message(reshade::log::level::warning,failure.what());return false;}
}
Json risky_panel_signature(const Json& settings) {
    Json result = Json::object();
    for (const auto& item : settings.items()) {
        const std::string key = item.key();
        if (key.starts_with("Model") || key.starts_with("Resolution") || key.starts_with("Block") ||
            key.starts_with("Chain") || key.starts_with("Fusion") || key.starts_with("Motion") ||
            key == "NvidiaDllFolder" || key == "ExecutionDevice" || key == "WorkerEngine" ||
            key == "Upscaler" || key == "AdditionalScaler" || key == "RefinementPasses" ||
            key == "ProcessingScale" || key == "ProcessingEnabled" || key == "HookStage")
            result[key] = item.value();
    }
    return result;
}
void record_recovery_failure(const char* text) noexcept {
    auto& value = state();
    if (!value.runtime_recovery || !value.runtime_recovery->pending()) return;
    try {
        if (value.runtime_recovery->failure(text))
            value.recovery_notice = "Initialization failed. Last working settings will be restored at startup.";
    } catch (const std::exception& failure) {
        reshade::log::message(reshade::log::level::warning, failure.what());
    }
}
void save_live_controls(reshade::api::effect_runtime* runtime) {
    if (!runtime)
        return;
    auto& addon_state = state();
    if (addon_state.runtime_recovery && addon_state.panel.pipeline.recovery) {
        try {
            const auto candidate = panel_json(addon_state.panel);
            const auto& report = addon_state.runtime_recovery->report();
            const auto& prior = addon_state.runtime_recovery->pending() ? report.at("pending") : report.at("working");
            if (risky_panel_signature(candidate) != risky_panel_signature(prior)) {
                addon_state.runtime_recovery->begin(candidate);
                addon_state.recovery_completed = addon_state.completed_retired;
                if (addon_state.direct_runner) addon_state.recovery_completed += addon_state.direct_runner->completed();
                if (addon_state.endpoint.session) addon_state.recovery_completed += addon_state.endpoint.session->completed();
            }
        } catch (const std::exception& recovery_error) {
            addon_state.recovery_notice = recovery_error.what();
            reshade::log::message(reshade::log::level::warning, recovery_error.what());
        }
    }
    phoenix::visit_settings(addon_state.panel, [&](const char* key, auto& value) {
        save_phoenix_value(runtime, "Phoenix", key, value);
    });
    for (unsigned iteration_index = 0; iteration_index < 4; ++iteration_index) {
        const auto section = "Phoenix.Preset" + std::to_string(iteration_index + 1);
        phoenix::visit_settings(addon_state.presets[iteration_index],
                                [&](const char* key, auto& value) {
                                    save_phoenix_value(runtime, section.c_str(), key, value);
                                });
    }
    for (unsigned iteration_index = 0; iteration_index < addon_state.hotkeys.size();
         ++iteration_index) {
        const auto key = "Hotkey" + std::to_string(iteration_index);
        reshade::set_config_value(runtime, "Phoenix", key.c_str(),
                                  addon_state.hotkeys[iteration_index]);
    }
}
void load_live_controls(reshade::api::effect_runtime* runtime) {
#if !CNR_PHOENIX_DIAGNOSTIC
    auto& addon_state = state();
    if (!runtime || addon_state.live_loaded)
        return;
    addon_state.live_loaded = true;
    phoenix::Settings saved;
    saved.worker_engine =
        addon_state.panel.worker_engine; // Legacy deployment default; saved selection wins.
    saved.nvidia_dll_folder = addon_state.panel.nvidia_dll_folder;
    saved.model_library_directory = addon_state.panel.model_library_directory;
    phoenix::visit_settings(saved, [&](const char* key, auto& value) {
        load_phoenix_value(runtime, "Phoenix", key, value);
    });
    if (saved.valid()) {
        const bool recovery_file=[]()noexcept{try{std::error_code error;return std::filesystem::is_regular_file(phoenix_pipeline::application_recovery_path(),error);}catch(...){return false;}}();
        if (saved.pipeline.recovery || recovery_file) {
            try {
                auto safe = addon_state.panel; safe.mode = phoenix::input; safe.uplift = false;
                addon_state.runtime_recovery = std::make_unique<phoenix_pipeline::RecoveryJournal>(
                    phoenix_pipeline::application_recovery_path(), panel_json(safe));
                if (const auto previous = saved.pipeline.recovery?addon_state.runtime_recovery->recover_interrupted():std::optional<Json>{}) {
                    if (!restore_panel_json(*previous, saved)) saved = safe;
                    addon_state.recovery_notice = "Restored last working settings after an interrupted initialization.";
                    phoenix::visit_settings(saved, [&](const char* key, auto& value) {
                        save_phoenix_value(runtime, "Phoenix", key, value);
                    });
                }
                if(const auto requested=addon_state.runtime_recovery->take_requested()){
                    if(!restore_panel_json(*requested,saved))saved=safe;
                    addon_state.recovery_notice="Applied recovery settings requested through DLSS Everything.";
                    phoenix::visit_settings(saved,[&](const char* key,auto& value){save_phoenix_value(runtime,"Phoenix",key,value);});
                }
                if(saved.pipeline.recovery)addon_state.runtime_recovery->begin(panel_json(saved));
                else addon_state.runtime_recovery.reset();
            } catch (const std::exception& recovery_error) {
                addon_state.recovery_notice = recovery_error.what();
                reshade::log::message(reshade::log::level::warning, recovery_error.what());
            }
        }
        addon_state.panel = saved;
        addon_state.timing_settings_dirty = true;
        addon_state.screenshot_request_seen = addon_state.panel.screenshot_request;
    } else
        reshade::log::message(reshade::log::level::warning,
                              "Phoenix DLSS5 ignored invalid saved panel settings");
    if (addon_state.configured && phoenix::uses_ours(addon_state.panel.mode) &&
        !addon_state.bundle && !addon_state.direct_runner && !addon_state.endpoint.session) {
        addon_state.weight_source = addon_state.configured_weight_source;
        // The selected source becomes active only before runner initialization.
        // Model validation reports invalid paths without changing a running model.
        try {
            addon_state.weight_source = phoenix_model_tools::selection(addon_state.panel, addon_state.configured_weight_source);
        } catch (const std::exception& feature_entry) {
            const auto fallback = addon_state.runtime_recovery
                ? addon_state.runtime_recovery->failure(feature_entry.what()) : std::optional<Json>{};
            if (fallback && restore_panel_json(*fallback, addon_state.panel)) {
                // Model selection failed before GPU initialization. Restore now,
                // rather than repeatedly trying the same invalid source.
                if (phoenix::uses_ours(addon_state.panel.mode)) {
                    try {
                        addon_state.weight_source = phoenix_model_tools::selection(addon_state.panel, addon_state.configured_weight_source);
                    } catch (...) { addon_state.panel.mode = phoenix::input; addon_state.panel.uplift = false; }
                }
                addon_state.recovery_notice = "Model initialization failed; restored last working settings.";
                phoenix::visit_settings(addon_state.panel, [&](const char* key, auto& value) {
                    save_phoenix_value(runtime, "Phoenix", key, value);
                });
                reshade::log::message(reshade::log::level::warning, addon_state.recovery_notice.c_str());
            } else {
                addon_state.terminal = true;
                error(feature_entry.what());
            }
        }
        addon_state.active_model_source = addon_state.panel.model_source;
        addon_state.active_model_folder = addon_state.panel.model_folder;
        addon_state.active_model_dll = addon_state.panel.model_dll;
        addon_state.active_model_id = addon_state.panel.model_id;
        addon_state.active_nvidia_dll_folder = addon_state.panel.nvidia_dll_folder;
    }
    for (unsigned iteration_index = 0; iteration_index < 4; ++iteration_index) {
        phoenix::Settings preset;
        preset.mode = int(iteration_index);
        const auto section = "Phoenix.Preset" + std::to_string(iteration_index + 1);
        phoenix::visit_settings(preset, [&](const char* key, auto& value) {
            load_phoenix_value(runtime, section.c_str(), key, value);
        });
        addon_state.presets[iteration_index].mode = int(iteration_index);
        if (preset.valid())
            addon_state.presets[iteration_index] = preset;
    }
    auto keys = addon_state.hotkeys;
    for (unsigned iteration_index = 0; iteration_index < keys.size(); ++iteration_index) {
        const auto key = "Hotkey" + std::to_string(iteration_index);
        reshade::get_config_value(runtime, "Phoenix", key.c_str(), keys[iteration_index]);
    }
    bool valid_keys = true;
    for (unsigned iteration_index = 0; iteration_index < keys.size(); ++iteration_index) {
        valid_keys &= keys[iteration_index] >= 0 && keys[iteration_index] <= 255;
        for (unsigned inner_index = 0; inner_index < iteration_index; ++inner_index)
            valid_keys &= keys[iteration_index] == 0 || keys[iteration_index] != keys[inner_index];
    }
    if (valid_keys)
        addon_state.hotkeys = keys;
#else
    (void)runtime;
#endif
}

std::filesystem::path overrides_path() {
    wchar_t filename[32768]{};
    const auto count = GetModuleFileNameW(state().self, filename, DWORD(std::size(filename)));
    if (!count || count >= std::size(filename))
        throw std::runtime_error("Phoenix DLSS5 addon path unavailable");
    return std::filesystem::path(filename).parent_path() / "PhoenixOverrides.ini";
}
void poll_overrides(reshade::api::effect_runtime* runtime, bool force = false) {
    auto& addon_state = state();
    const auto now = GetTickCount64();
    if (!force && (!addon_state.panel.external_reload || now - addon_state.override_tick < 1000))
        return;
    addon_state.override_tick = now;
    try {
        const auto path = overrides_path();
        if (!std::filesystem::exists(path)) {
            addon_state.override_status = "No PhoenixOverrides.ini next to addon";
            return;
        }
        const auto mtime = std::filesystem::last_write_time(path);
        if (!force && addon_state.override_seen && mtime == addon_state.override_mtime)
            return;
        addon_state.override_mtime = mtime;
        addon_state.override_seen = true;
        if (std::filesystem::file_size(path) > 65536)
            throw std::runtime_error("Override file exceeds 64 KiB");
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("Override file cannot be opened");
        std::string content(65537, '\0');
        input.read(content.data(), std::streamsize(content.size()));
        content.resize(size_t(input.gcount()));
        unsigned applied = 0, ignored = 0;
        auto next = addon_state.panel;
        if (!phoenix::merge_overrides(content, next, applied, ignored))
            throw std::runtime_error("Override settings rejected; current settings retained");
        if (applied) {
            auto previous = panel_json(addon_state.panel), incoming = panel_json(next);
            for (const char* key :
                 {"ScreenshotRequest", "CaptureBurst", "ExternalReload", "LogLevel"}) {
                previous.erase(key);
                incoming.erase(key);
            }
            const bool render_changed = previous != incoming;
            addon_state.panel = next;
            if (render_changed) {
                addon_state.live.reset_requested = true;
                addon_state.timing_settings_dirty = true;
            }
            ++addon_state.override_updates;
            save_live_controls(runtime);
            record_event("phoenix_overrides", {{"applied", applied}, {"ignored", ignored}});
            {
                char msg[224]{};
                snprintf(
                    msg, sizeof(msg),
                    "[Phoenix DLSS5] overrides: %u applied, %u ignored; backend=%d comparison=%d sources=%d,%d intensity=%.2f",
                    applied, ignored, addon_state.panel.mode, addon_state.panel.comparison,
                    addon_state.panel.source_a, addon_state.panel.source_b,
                    double(addon_state.panel.intensity));
                reshade::log::message(reshade::log::level::info, msg);
            }
        }
        addon_state.override_status =
            std::to_string(applied) + " fields applied, " + std::to_string(ignored) + " ignored";
    } catch (const std::exception& feature_entry) {
        addon_state.override_status = feature_entry.what();
        reshade::log::message(reshade::log::level::warning, feature_entry.what());
    }
}
// Offline input-capture burst: fence-gated snapshots of the owned color /
// motion inputs plus the presented output, with a key=value meta sidecar
// for the replay probe. Bursts up to 64 full frames (~1.6 GB worst case)
// hitch the game while flushing to disk; captured content itself is
// unaffected.
void arm_input_capture(unsigned frames) {
    auto& addon_state = state();
    if (!frames || !addon_state.direct_runner || addon_state.direct_runner->capture_armed())
        return;
    frames = std::min(frames, 64u);
    std::error_code ec;
    std::filesystem::create_directories(addon_state.output_directory / "captures", ec);
    if (ec) {
        reshade::log::message(
            reshade::log::level::warning,
            ("[Phoenix DLSS5] input capture: cannot create captures dir: " + ec.message()).c_str());
        return;
    }
    char base[128]{};
    snprintf(base, sizeof(base), "captures/phx_%llu_m%u", (unsigned long long)GetTickCount64(),
             unsigned(addon_state.panel.mode));
    addon_state.capture_base = base;
    addon_state.direct_runner->arm_capture(addon_state.capture_base, frames);
    reshade::log::message(reshade::log::level::info,
                          ("[Phoenix DLSS5] input capture armed: " + addon_state.capture_base +
                           " x" + std::to_string(frames))
                              .c_str());
}
void write_capture_frame(phoenix::direct::CaptureFrame& cap) {
    auto& addon_state = state();
    char stem[192]{};
    snprintf(stem, sizeof(stem), "%s_f%04u", addon_state.direct_runner->capture_base().c_str(),
             cap.meta.frame);
    auto put = [&](const char* ext, const std::vector<uint8_t>& bytes) {
        if (bytes.empty())
            return;
        std::ofstream out(addon_state.output_directory / (std::string(stem) + ext),
                          std::ios::binary | std::ios::trunc);
        if (out)
            out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    };
    put(".color.bin", cap.color);
    put(".motion.bin", cap.motion);
    put(".output.bin", cap.output);
    if(cap.meta.appearance_flags) {
        const char* names[]={".mask.rgba32f",".ui.rgba32f",".alpha.r32f",".backbuffer.rgba16f",".distortion.rgba32f"};
        Json planes=Json::object();
        for(unsigned i=0;i<5;++i)if(!cap.appearance[i].empty()) {
            put(names[i],cap.appearance[i]);
            planes[names[i]]={{"bytes",cap.appearance[i].size()},
                {"sha256",phoenix_model::weight_digest(cap.appearance[i].data(),cap.appearance[i].size())}};
        }
        std::ofstream out(addon_state.output_directory/(std::string(stem)+".appearance.json"));
        if(out)out<<Json({{"schema","phoenix-appearance-capture/1"},{"profile","phoenix-experimental-rgb/1"},
            {"generation",cap.meta.appearance_generation},{"frame",cap.meta.frame},
            {"width",cap.meta.width},{"height",cap.meta.height},{"planes",planes}}).dump(2);
    }
    if (!cap.lineage[0].empty()) {
        const char* names[]{".proxy.bin", ".native.bin", ".phoenix.bin", ".feed.bin"};
        for (unsigned iteration_index = 0; iteration_index < 4; ++iteration_index)
            put(names[iteration_index], cap.lineage[iteration_index]);
        put(".exposure.bin", cap.exposure);
        std::ofstream out(addon_state.output_directory / (std::string(stem) + ".lineage.json"));
        if (out)
            out << Json(
                       {{"format", "RGBA16F"},
                        {"source_a", cap.source_a},
                        {"source_b", cap.source_b},
                        {"divider", cap.divider},
                        {"reference_feed", cap.meta.reference_feed},
                        {"srgb", cap.meta.srgb},
                        {"scope",
                         "Same-frame proxy, DLL output, Phoenix DLSS5 output, Phoenix DLSS5 feed, and final engine output"}})
                       .dump(2);
    }
    {
        std::ofstream out(addon_state.output_directory / (std::string(stem) + ".meta"));
        if (out)
            out << phoenix::direct::capture_meta_write(cap.meta);
    }
    ++addon_state.captures_taken;
    auto meanabs = [](const std::vector<uint8_t>& bytes) {
        if (bytes.empty() || (bytes.size() % 4) != 0)
            return -1.0;
        double absolute_sum = 0;
        const size_t float_count = bytes.size() / 4;
        for (size_t iteration_index = 0; iteration_index < float_count; ++iteration_index) {
            float sample_value = 0;
            std::memcpy(&sample_value, bytes.data() + iteration_index * 4, 4);
            absolute_sum += double(sample_value < 0 ? -sample_value : sample_value);
        }
        return absolute_sum / double(float_count);
    };
    char stat[192]{};
    snprintf(stat, sizeof(stat),
             "[Phoenix DLSS5] input capture wrote %s cin=%.5f mot=%.5f cout=%.5f", stem,
             meanabs(cap.color), meanabs(cap.motion), meanabs(cap.output));
    reshade::log::message(reshade::log::level::info, stat);
}
// ReShade owns screenshot encoding/naming; route its output before requesting it.
// Compare before setting: set_config_value persists the INI, so never write per frame.
void route_screenshots(reshade::api::effect_runtime* runtime) {
    const auto& folder = state().output_directory;
    if (!runtime || folder.empty())
        return;
    const auto destination = phoenix_model::source_path_string(folder);
    char current[32768]{};
    size_t size = sizeof(current);
    if (!reshade::get_config_value(runtime, "SCREENSHOT", "SavePath", current, &size) ||
        destination != current)
        reshade::set_config_value(runtime, "SCREENSHOT", "SavePath", destination.c_str());
}
// Commands run on ReShade's runtime callback under State::mutex, never on a
// detached thread. GPU fences remain owned by the existing runner.
void bridge_write(const std::filesystem::path& path, const Json& value) {
    auto temp = path;
    temp += L".tmp";
    {
        std::ofstream file(temp, std::ios::binary | std::ios::trunc);
        check(bool(file), "Bridge response cannot be opened");
        file << value.dump(2);
        file.flush();
        check(bool(file), "Bridge response write failed");
    }
    check(MoveFileExW(temp.c_str(), path.c_str(),
                      MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0,
          "Bridge atomic publication failed");
}
Json bridge_snapshot() {
    auto& addon_state = state();
    Json devices = Json::array();
    for (const auto& device_info : addon_state.execution_devices)
        devices.push_back({{"id", device_info.id}, {"label", device_info.label}});
    return {
        {"tick_ms", GetTickCount64()},
        {"settings", panel_json(addon_state.panel)},
        {"execution_devices", devices},
        {"presentation_scope", "DLSS output texture; final game display is not independently verified"},
        {"nvapi", phoenix_native::Api::lifetime_report()},
        {"model_source",
         {{"loaded_model_id", addon_state.weight_source.model_id},
          {"switch_error", addon_state.model_switch_error},
          {"restart_required", false},
          {"kind", addon_state.weight_source.disk_model()  ? "model_folder"
                   : addon_state.weight_source.dll.empty() ? "files"
                                                           : "nvidia_dll"},
          {"path", phoenix_model::source_path_string(addon_state.weight_source.disk_model() ? addon_state.weight_source.folder
                                                           : addon_state.weight_source.dll)},
          {"reload_pending",
           addon_state.panel.nvidia_dll_folder != addon_state.active_nvidia_dll_folder ||
           addon_state.panel.model_source != addon_state.active_model_source ||
               (addon_state.panel.model_folder != addon_state.active_model_folder ||
                (addon_state.panel.model_dll != addon_state.active_model_dll || addon_state.panel.model_id != addon_state.active_model_id))}}},
        {"embedded_runtime",
         {{"identity", addon_state.embedded.identity},
          {"files", addon_state.embedded.files},
          {"root", phoenix_model::source_path_string(addon_state.embedded.root)}}},
        {"model_validation",
         {{"busy", addon_state.model_validation.poll().busy},
          {"passed", addon_state.model_validation.poll().passed},
          {"message", addon_state.model_validation.poll().message}}},
        {"input", addon_state.bridge_inputs},
        {"input_tick_ms", addon_state.bridge_input_tick},
        {"phoenix",
         {{"status", addon_state.direct_status},
          {"calls", addon_state.direct_calls},
          {"deferred", addon_state.direct_deferred},
          {"busy", addon_state.direct_busy},
          {"recorded_frames", addon_state.recorded},
          {"submitted_frames", addon_state.submitted}}},
        {"nvidia",
         {{"status", addon_state.direct_nr_status},
          {"ready", addon_state.direct_nr_ready},
          {"feature", addon_state.selfdrive.feature_id},
          {"creates", addon_state.selfdrive.creates},
          {"evaluates", addon_state.selfdrive.evaluates},
          {"create_result", addon_state.selfdrive.last_create_result},
          {"evaluate_result", addon_state.selfdrive.last_evaluate_result}}},
        {"execution", addon_state.direct_runner ? addon_state.direct_runner->execution_status()
                                                : Json{{"kind", "not_started"}}},
        {"worker_timing", addon_state.worker_timing.report()},
        {"capture",
         {{"armed", addon_state.direct_runner && addon_state.direct_runner->capture_armed()},
          {"frames_written", addon_state.captures_taken},
          {"readback_bytes",
           addon_state.direct_runner ? addon_state.direct_runner->capture_readback_bytes() : 0},
          {"base", addon_state.capture_base}}},
        {"present_fps", addon_state.present_fps},
        {"gpu_timing", addon_state.live_timing ? addon_state.live_timing->report() : Json(nullptr)},
        {"scope",
         "Recorded frames and command acknowledgement do not prove completed rendering or visual correctness"}};
}
void screenshot_saved(reshade::api::effect_runtime*, const char* path) {
    if(!path||!*path)return;
    try {
        auto& addon_state=state();
        std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
        if(!addon_state.panel.pipeline.screenshot_receipt)return;
        const auto image=std::filesystem::u8path(path);
        auto facts=bridge_snapshot();
        facts["application"]="Phoenix addon";
        facts["model_identity"]=addon_state.weight_source.configuration();
        facts["frame_identity_scope"]="State observed when ReShade reports the saved image; exact source-frame association is unavailable";
        const auto receipt=phoenix_pipeline::save_screenshot_receipt(image,
            phoenix_model::mapped_file_digest(image),std::move(facts));
        reshade::log::message(reshade::log::level::info,
            ("[Phoenix DLSS5] screenshot receipt: "+phoenix_model::source_path_string(receipt)).c_str());
    }catch(const std::exception& error){
        reshade::log::message(reshade::log::level::warning,
            (std::string("[Phoenix DLSS5] screenshot saved, receipt failed: ")+error.what()).c_str());
    }
}
void poll_control_bridge(reshade::api::effect_runtime* runtime) {
    auto& addon_state = state();
    const auto now = GetTickCount64();
    if (addon_state.output_directory.empty() || now - addon_state.bridge_poll_tick < 250)
        return;
    addon_state.bridge_poll_tick = now;
    const auto folder = addon_state.output_directory / "bridge";
    try {
        if (!addon_state.bridge_endpoint_written) {
            std::filesystem::create_directories(folder);
            if (addon_state.bridge_session.empty()) {
                FILETIME created{}, exited{}, kernel{}, user{};
                check(GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user) != 0,
                      "Bridge process identity unavailable");
                addon_state.bridge_session =
                    std::to_string(GetCurrentProcessId()) + "-" +
                    std::to_string((uint64_t(created.dwHighDateTime) << 32) |
                                   created.dwLowDateTime);
            }
            bridge_write(folder / "endpoint.json",
                         {{"schema", "phoenix-control/1"},
                          {"session", addon_state.bridge_session},
                          {"pid", GetCurrentProcessId()},
                          {"operations",
                           {"status", "set", "reset_history", "screenshot", "capture", "save",
                            "validate_model"}}});
            addon_state.bridge_endpoint_written = true;
        }
        if (!addon_state.bridge_pending_response.is_null()) {
            bridge_write(folder / "response.json", addon_state.bridge_pending_response);
            addon_state.bridge_pending_response = nullptr;
        }
        const auto incoming = folder / "request.json", claimed = folder / "processing.json";
        if (!std::filesystem::exists(incoming))
            return;
        // Atomic claim prevents re-execution after a response write failure.
        check(MoveFileExW(incoming.c_str(), claimed.c_str(), MOVEFILE_REPLACE_EXISTING) != 0,
              "Bridge request claim failed");
        Json response{{"schema", "phoenix-control/1"},
                      {"session", addon_state.bridge_session},
                      {"pid", GetCurrentProcessId()},
                      {"id", nullptr},
                      {"ok", false}};
        try {
            check(std::filesystem::file_size(claimed) <= 65536, "Bridge request exceeds 64 KiB");
            std::ifstream file(claimed, std::ios::binary);
            std::string bytes(65537, '\0');
            file.read(bytes.data(), std::streamsize(bytes.size()));
            bytes.resize(size_t(file.gcount()));
            check(bytes.size() <= 65536, "Bridge request exceeds 64 KiB");
            const auto request = Json::parse(bytes);
            if (request.contains("id") && request["id"].is_string() &&
                request["id"].get_ref<const std::string&>().size() <= 64)
                response["id"] = request["id"];
            const auto cmd = phoenix::control_bridge::validate(
                request, addon_state.bridge_session, GetCurrentProcessId(), now, addon_state.panel);
            check(!cmd.enabled && !cmd.coherent_frames, "Frontend controls require a compatibility session");
            check(cmd.id != addon_state.bridge_last_id,
                  "Duplicate bridge request; inspect previous response");
            addon_state.bridge_last_id = cmd.id;
            std::string phase = "observed";
            if (cmd.op == "set") {
                auto before = panel_json(addon_state.panel), after = panel_json(cmd.settings);
                for (const auto* key :
                     {"ScreenshotRequest", "CaptureBurst", "ExternalReload", "LogLevel"}) {
                    before.erase(key);
                    after.erase(key);
                }
                addon_state.panel = cmd.settings;
                if (before != after) {
                    addon_state.live.reset_requested = true;
                    addon_state.timing_settings_dirty = true;
                }
                save_live_controls(runtime);
                phase = "applied";
            } else if (cmd.op == "save") {
                save_live_controls(runtime);
                addon_state.settings_saved_at = now;
                phase = "saved";
            } else if (cmd.op == "validate_model") {
                check(!addon_state.model_validation.poll().busy,
                      "Model validation is already running");
                addon_state.model_validation.validate(
                    phoenix_model_tools::selection(addon_state.panel, addon_state.configured_weight_source),
                    phoenix::portable::native_runtime_root());
                phase = "queued";
            } else if (cmd.op == "reset_history") {
                addon_state.live.reset_requested = true;
                phase = "queued";
            } else if (cmd.op == "screenshot") {
                route_screenshots(runtime);
                const auto name = "Phoenix-bridge-" + cmd.id;
                runtime->save_screenshot(name.c_str());
                response["screenshot_postfix"] = name;
                phase = "queued";
            } else if (cmd.op == "capture") {
                check(addon_state.direct_runner != nullptr,
                      "Capture unavailable: no direct runner");
                check(!addon_state.direct_runner->capture_armed(), "A capture is already armed");
                arm_input_capture(cmd.frames);
                check(addon_state.direct_runner->capture_armed(), "Capture could not be armed");
                phase = "queued";
            }
            response["ok"] = true;
            response["phase"] = phase;
        } catch (const std::exception& feature_entry) {
            response["error"] = feature_entry.what();
            response["phase"] = "rejected";
        }
        response["snapshot"] = bridge_snapshot();
        addon_state.bridge_pending_response = response;
        bridge_write(folder / "response.json", addon_state.bridge_pending_response);
        addon_state.bridge_pending_response = nullptr;
        addon_state.bridge_status = response.value("phase", "");
        std::error_code ec;
        std::filesystem::remove(claimed, ec);
    } catch (const std::exception& feature_entry) {
        addon_state.bridge_status = feature_entry.what();
    }
}
void live_runtime_tick(reshade::api::effect_runtime* runtime) {
    if(phoenix_compat::applies(runtime))return;
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    load_live_controls(runtime);
    addon_state.present_runtime = runtime;
    phoenix_ui::sanitize_bindings(addon_state.hotkeys);
    if (runtime && !addon_state.output_directory.empty() &&
        addon_state.screenshot_runtime != runtime) {
        route_screenshots(runtime);
        addon_state.screenshot_runtime = runtime;
    }
    if (runtime && addon_state.direct_capture && !addon_state.direct_capture_done &&
        addon_state.recorded >= addon_state.next_capture_at) {
        route_screenshots(runtime);
        runtime->save_screenshot("Phoenix-direct");
        if (addon_state.capture_every > 0)
            addon_state.next_capture_at += addon_state.capture_every;
        else
            addon_state.direct_capture_done = true;
    }
#if !CNR_PHOENIX_DIAGNOSTIC
    if (!runtime)
        return;
    poll_overrides(runtime);
    poll_control_bridge(runtime);
    menu_helper::tick(addon_state.sandbox, addon_state.output_directory);
    if (!addon_state.output_directory.empty() && addon_state.panel.screenshot_request > 0 &&
        addon_state.panel.screenshot_request != addon_state.screenshot_request_seen) {
        addon_state.screenshot_request_seen = addon_state.panel.screenshot_request;
        const auto postfix =
            "Phoenix-request-" + std::to_string(addon_state.screenshot_request_seen);
        Json settings = Json::object();
        phoenix::visit_settings(addon_state.panel,
                                [&](const char* key, auto& value) { settings[key] = value; });
        const Json capture{{"request", addon_state.screenshot_request_seen},
                           {"recorded_frames", addon_state.recorded},
                           {"native_evaluates", addon_state.selfdrive.evaluates},
                           {"native_feature", addon_state.selfdrive.feature_id},
                           {"native_ready", addon_state.direct_nr_ready},
                           {"native_status", addon_state.direct_nr_status},
                           {"settings", settings},
                           {"gpu_timing", addon_state.live_timing
                                              ? addon_state.live_timing->report()
                                              : Json(nullptr)},
                           {"scope", "Screenshot requested; verify ReShade image file separately"}};
        std::ofstream sidecar(addon_state.output_directory / (postfix + ".json"));
        if (sidecar)
            sidecar << capture.dump(2);
        route_screenshots(runtime);
        runtime->save_screenshot(postfix.c_str());
        reshade::log::message(reshade::log::level::info,
                              ("[Phoenix DLSS5] screenshot requested: " + postfix).c_str());
    }
    // INI-driven auto-arm waits until past the black loading frames; manual
    // hotkey/button arming bypasses this gate and fires immediately in-game.
    if (addon_state.panel.capture_burst > 0 && addon_state.direct_runner &&
        !addon_state.direct_runner->capture_armed() &&
        addon_state.recorded >= addon_state.direct_capture_frame) {
        arm_input_capture(unsigned(addon_state.panel.capture_burst));
        addon_state.panel.capture_burst = 0;
        save_live_controls(runtime);
    }
    if (addon_state.capture_hotkey >= 0) {
        if (GetTickCount64() >= addon_state.hotkey_deadline) {
            addon_state.capture_hotkey = -1;
            return;
        }
        const auto key = runtime->last_key_pressed();
        if (!key)
            return;
        const auto result = phoenix_ui::capture_binding(addon_state.hotkeys, addon_state.capture_hotkey, key);
        if (result == phoenix_ui::BindingResult::Accepted || result == phoenix_ui::BindingResult::Cancelled) {
            addon_state.capture_hotkey = -1;
            save_live_controls(runtime);
        } else addon_state.error = phoenix_ui::binding_error(result);
        return;
    }
    if (addon_state.hotkeys[0] && runtime->is_key_pressed(unsigned(addon_state.hotkeys[0]))) {
        addon_state.panel.mode = (addon_state.panel.mode + 1) % 4;
        addon_state.live.reset_requested = true;
        save_live_controls(runtime);
    }
    if (addon_state.hotkeys[1] && runtime->is_key_pressed(unsigned(addon_state.hotkeys[1])))
        addon_state.live.reset_requested = true;
    if (addon_state.hotkeys[2] && runtime->is_key_pressed(unsigned(addon_state.hotkeys[2]))) {
        ++addon_state.markers;
        record_event("phoenix_marker", {{"index", addon_state.markers},
                                        {"mode", addon_state.panel.mode},
                                        {"frame", addon_state.routed_frames}});
    }
    if (addon_state.hotkeys[3] && runtime->is_key_pressed(unsigned(addon_state.hotkeys[3])) &&
        addon_state.panel.mode == phoenix::ours && addon_state.panel.experimental) {
        addon_state.panel.transformer_blocks = addon_state.panel.transformer_blocks == 1
                                                   ? 8
                                                   : addon_state.panel.transformer_blocks - 1;
        addon_state.panel.minimum_blocks =
            std::min(addon_state.panel.minimum_blocks, addon_state.panel.transformer_blocks);
        addon_state.live.reset_requested = true;
        save_live_controls(runtime);
    }
    if (addon_state.hotkeys[4] && runtime->is_key_pressed(unsigned(addon_state.hotkeys[4])) &&
        addon_state.live_timing) {
        addon_state.live_timing->restart();
        addon_state.worker_timing.clear();
    }
    if (addon_state.hotkeys[5] && runtime->is_key_pressed(unsigned(addon_state.hotkeys[5])))
        arm_input_capture(
            addon_state.panel.capture_burst > 0 ? unsigned(addon_state.panel.capture_burst) : 8u);
#endif
}

phoenix_ui::Context make_panel_context(reshade::api::effect_runtime* runtime,
                                      const std::filesystem::path& model_path, bool disk_model) {
    auto& value = state();
    phoenix_ui::Context ui{value.panel, value.live, value.presets, value.hotkeys,
                           value.capture_hotkey, value.hotkey_deadline, value.model_picker,
                           value.execution_devices, value.error, value.markers};
    ui.status = phoenix::direct::startup_status(value.configured, value.automatic, gpu_gate(), value.direct_game,
        bool(value.endpoint.session) || (value.direct_runner && value.direct_runner->submitted() > 0),
        value.terminal, value.observe);
    ui.processing_active = !value.terminal && value.last_mode >= 0 &&
        (bool(value.endpoint.session) || (value.direct_runner && value.direct_runner->submitted() > 0));
    ui.active_route = value.direct_runner && value.direct_runner->external()
        ? "Portable worker / game DLSS textures" : "Native D3D12 / game DLSS textures";
    if (value.direct_game) ui.status += " | " + value.direct_status;
    ui.active_device = value.direct_runner && value.direct_runner->external()
        ? "Worker device (details in Performance)" : "Game graphics device";
    ui.direct_game = value.direct_game;
    ui.host_controls_stage = value.direct_feeder;
    ui.direct_nr_ready = value.direct_nr_ready;
    ui.nvidia_status = value.direct_nr_status;
    ui.appearance=value.bridge_inputs.value("appearance",Json::object());
    ui.portable_backend_selected=value.panel.execution_device!="native";
    if(value.direct_runner&&value.direct_runner->external()){
        const auto execution=value.direct_runner->execution_status();
        const auto geometry=value.direct_runner->resolution();
        ui.portable_chain_available=execution.value("last_result",Json::object()).value("portable_pipeline_contract",0)>=1&&
            geometry.width==geometry.output_width&&geometry.height==geometry.output_height;
    }
    ui.model_path = model_path;
    ui.disk_model = disk_model;
    ui.runtime_root = phoenix::portable::native_runtime_root();
    std::error_code file_error;
    ui.runtime_available = std::filesystem::is_regular_file(ui.runtime_root / "phoenix_runtime_cpp.dll", file_error)
                           && std::filesystem::is_regular_file(ui.runtime_root / "phoenix_native_cpp.json", file_error)
                           && std::filesystem::is_regular_file(model_path, file_error);
    ui.bundle_path = value.bundle_path;
    ui.overrides_path = overrides_path();
    ui.configured_weight_source = value.configured_weight_source;
    ui.active_model_source = value.active_model_source;
    ui.active_model_folder = value.active_model_folder;
    ui.active_model_dll = value.active_model_dll;
    ui.active_nvidia_dll_folder = value.active_nvidia_dll_folder;
    ui.active_model_id = value.active_model_id;
    ui.loaded_model_id = value.direct_runner && phoenix::uses_ours(value.panel.mode)
        ? value.weight_source.model_id : std::string{};
    ui.model_restart = value.model_switch_error.empty()
        ? "Model change pending; it will load after the current GPU frame finishes."
        : value.model_switch_error;
    ui.model_switch_error = value.model_switch_error;
    ui.override_status = value.override_status;
    ui.last_transformer_blocks = value.last_transformer_blocks;
    ui.last_mode = value.last_mode;
    ui.present_fps = value.present_fps;
    if (value.live_timing) {
        ui.live_timing.emplace();
        ui.live_timing->data = value.live_timing->report();
        const auto& history = value.live_timing->history();
        const auto count = value.live_timing->history_count();
        const auto offset = value.live_timing->history_offset();
        for (size_t index = 0; index < count; ++index)
            ui.live_timing->samples.push_back(history[(offset + index) % history.size()]);
    }
    ui.worker_timing.data = value.worker_timing.report();
    ui.depth_budget.estimate = value.depth_budget.ema();
    ui.live_resets_recorded = value.live_resets_recorded;
    ui.switches = value.switches;
    ui.mode_frames = value.mode_frames;
    ui.mode_own = value.mode_own;
    ui.mode_native = value.mode_native;
    if (value.direct_runner)
        ui.direct_runner = phoenix_ui::Preview{value.direct_runner->resolution(), value.direct_runner->upscale_status()};
    ui.save = [runtime] { save_live_controls(runtime); };
    ui.capture = [](unsigned count) { arm_input_capture(count); };
    ui.reload = [runtime] { poll_overrides(runtime, true); };
    ui.marker = [] {
        auto& value = state();
        record_event("phoenix_marker", {{"index", value.markers}, {"mode", value.panel.mode},
                                        {"frame", value.routed_frames}});
    };
    ui.action = [](phoenix_ui::Action action) {
        auto& value = state();
        if (action == phoenix_ui::Action::ResetHistory) value.live.reset_requested = true;
        if (action == phoenix_ui::Action::ClearTiming) {
            if (value.live_timing) value.live_timing->restart();
            value.worker_timing.clear();
        }
    };
    ui.model_validation.poll = [] { return state().model_validation.poll(); };
    ui.model_validation.validate = [](const phoenix_model::WeightSource& source, const std::filesystem::path& root) {
        state().model_validation.validate(source, root);
    };
    return ui;
}

// Detailed route telemetry lives behind Performance, away from the play controls.
void draw_native_performance_details() {
    auto& addon_state = state();
    ImGui::Text("Phoenix DLSS5 frames submitted: %llu", addon_state.submitted);
    if (addon_state.direct_game) {
        ImGui::TextWrapped("%s", addon_state.direct_status.c_str());
        ImGui::TextDisabled("Output status tracks the DLSS texture; final display needs a visual check.");
    }
    const phoenix::PanelVisibility visible{addon_state.panel};
    ImGui::Text("Active: %s",
                addon_state.last_mode >= 0 ? phoenix::names[addon_state.last_mode] : "Waiting");

    if (addon_state.direct_runner && addon_state.direct_runner->external()) {
        const auto execution = addon_state.direct_runner->execution_status();
        const auto& result = execution.at("last_result");
        const auto source = execution.value("ready_frame", uint64_t(0));
        ImGui::Text("Worker: %s", execution.value("state", std::string("unknown")).c_str());
        const auto worker_error = execution.value("error", std::string());
        if (!worker_error.empty())
            ImGui::TextWrapped("Worker error: %s", worker_error.c_str());
        if (result.contains("kernel_engine"))
            ImGui::Text("Active execution method: %s",
                        result.value("kernel_engine", std::string()).c_str());
        if (result.contains("provider"))
            ImGui::Text("Active provider: %s", result.value("provider", std::string()).c_str());
        if (result.contains("error"))
            ImGui::TextWrapped("Last result error: %s",
                               result.value("error", std::string()).c_str());
        ImGui::Text("Worker elapsed: %.1f ms | results received: %llu",
                    result.value("elapsed_ms", 0.0), execution.value("delivered", uint64_t(0)));
        if (result.contains("phase_ms")) {
            const auto& phases = result.at("phase_ms");
            ImGui::Text("Prepare %.1f ms | Execute and transfers %.1f ms | Resolve %.1f ms",
                        phases.value("features_ms", 0.0), phases.value("graph_roundtrip_ms", 0.0),
                        phases.value("resolve_ms", 0.0));
        }
        if (result.contains("resident_memory"))
            ImGui::Text("Network GPU buffers: %.0f MiB (peak %.0f MiB)",
                        result.at("resident_memory").value("allocated_bytes", uint64_t(0)) /
                            1048576.0,
                        result.at("resident_memory").value("peak_allocated_bytes", uint64_t(0)) /
                            1048576.0);
        if (!source && result.value("ok", false) &&
            execution.value("last_discard_reason", std::string()) == "result_age_limit")
            ImGui::TextWrapped(
                "Last result arrived too late: %llu ms / %llu frames. Freshness limits are %d ms / %d frames.",
                execution.value("last_result_age_ms", uint64_t(0)),
                execution.value("last_result_age_frames", uint64_t(0)),
                addon_state.panel.offload_age_ms, addon_state.panel.offload_age_frames);
        const auto coverage = execution.value("composition", Json::object());
        if (!coverage.empty()) {
            ImGui::Text("Sampled output: %.1f%% changed | %.1f%% valid motion coverage",
                        coverage.value("changed_percent", 0.0),
                        coverage.value("valid_percent", 0.0));
            if (!coverage.value("changed_samples", 0u))
                ImGui::TextWrapped(
                    "No measured worker effect: the available result did not change sampled pixels.");
        }
        if (source)
            ImGui::Text("Available result age: %llu frames",
                        addon_state.direct_runner->submitted() - source);
        else
            ImGui::TextUnformatted("No fresh worker result; showing game output.");
    }
    // Recent GPU completions alongside the presentation rate. Lifetime means
    // stay in the detailed table; neither GPU interval measures a whole frame.
    if (addon_state.live_timing) {
        const auto perf = addon_state.live_timing->report();
        const char* own_txt = "--";
        const char* nat_txt = "--";
        char own_buf[64]{}, nat_buf[64]{};
        double route_ms = 0;
        for (const auto& row : perf.at("rows"))
            if (row.at("mode") == addon_state.last_mode &&
                row.at("transformer_blocks") == addon_state.last_transformer_blocks &&
                row.at("settings") == perf.at("active_settings")) {
                const auto& recent = row.at("recent");
                route_ms = recent.at("route_ms").get<double>();
                if (recent.at("phoenix_samples").get<uint64_t>()) {
                    snprintf(own_buf, sizeof own_buf, "%.3f ms (%llu)",
                             recent.at("phoenix_ms").get<double>(),
                             recent.at("phoenix_samples").get<uint64_t>());
                    own_txt = own_buf;
                }
                if (recent.at("nvidia_samples").get<uint64_t>()) {
                    snprintf(nat_buf, sizeof nat_buf, "%.3f ms (%llu)",
                             recent.at("nvidia_ms").get<double>(),
                             recent.at("nvidia_samples").get<uint64_t>());
                    nat_txt = nat_buf;
                }
                break;
            }
        ImGui::Text("Network GPU, recent: Phoenix DLSS5 %s | NVIDIA DLSS5 %s", own_txt, nat_txt);
        ImGui::Text("Addon GPU: %.3f ms | Game: %.1f FPS (%.2f ms between presents)", route_ms,
                    addon_state.present_fps,
                    addon_state.present_fps > 0 ? 1000.0 / addon_state.present_fps : 0);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "GPU times exclude CPU work and the rest of the game. FPS counts presentation callbacks, not displayed frames.\n"
                "Recent GPU samples retire asynchronously; presentation rate uses a separate one-second window.");
        if (addon_state.panel.log_level == 2)
            ImGui::TextUnformatted("Verbose tracing enabled: file writes can reduce FPS.");
    }
    if (visible.own_details() && addon_state.endpoint.session) {
        const auto controls = addon_state.endpoint.session->frame_controls();
        if (controls.contains("tone") && controls.contains("structure"))
            ImGui::Text("Last Phoenix DLSS5 tone %.2f | structure %.2f",
                        controls.at("tone").get<double>(), controls.at("structure").get<double>());
    }
    if (addon_state.timestamps && addon_state.endpoint.session) {
        const auto report = addon_state.timestamps->report();
        ImGui::Text("candidate %.3f ms (%llu samples)",
                    report.at("candidate_mean_ms").get<double>(),
                    report.at("candidate_samples").get<uint64_t>());
        if (ImGui::BeginTable("phoenix_groups", 3)) {
            for (const auto& group_timing : report.at("group_mean_ms")) {
                const unsigned group_index = group_timing.at("group").get<unsigned>();
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(group_index < kGroupCount ? kGroupLabels[group_index] : "?");
                ImGui::TableNextColumn();
                ImGui::Text("%llu", group_timing.at("samples").get<uint64_t>());
                ImGui::TableNextColumn();
                ImGui::Text("%.4f ms", group_timing.at("mean_ms").get<double>());
            }
            ImGui::EndTable();
        }
    } else if (!addon_state.live_timing) {
        ImGui::TextUnformatted("GPU profiling inactive");
    }

}

// UI and Evaluate share this lock; settings are immutable during recording.
void draw_phoenix_overlay(reshade::api::effect_runtime* runtime) {
    if (phoenix_compat::applies(runtime)) { phoenix_compat::panel(runtime); return; }
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    load_live_controls(runtime);
    const bool disk_model =
        addon_state.active_model_source == "model_folder" || addon_state.weight_source.disk_model();
    const auto model_path =
        disk_model
            ? addon_state.weight_source.folder / "model.json"
            : (!addon_state.weight_source.dll.empty()
                   ? addon_state.weight_source.dll
                   : ((addon_state.direct_game && addon_state.panel.execution_device != "native" &&
                       addon_state.model_dll_path.empty())
                          ? addon_state.portable_model_dll_path
                          : addon_state.model_dll_path));
    const auto model_status = addon_state.model_dll_monitor.poll(model_path, GetTickCount64());
    auto ui = make_panel_context(runtime, model_path, disk_model);
    ui.route_notice = addon_state.recovery_notice;
    if (phoenix::uses_ours(addon_state.panel.mode) && model_status != phoenix::ModelDllStatus::available) {
        ui.route_notice += (ui.route_notice.empty() ? "" : "\n") + std::string(disk_model
            ? "Cannot read model.json. Open Setup to choose and validate model files."
            : phoenix::model_dll_warning(model_status));
    }
    ui.setup_details = [&] {
#if defined(PHOENIX_EXPERIMENTAL_NATIVE)
        ImGui::TextWrapped("Experimental RTX compatibility build. RTX 20/30/40 execution needs testing on those cards; RTX 5080 checks are listed in the package notes.");
#endif
        if (!model_path.empty()) ImGui::TextWrapped("Configured path: %s", phoenix_model::source_path_string(model_path).c_str());
        ImGui::TextWrapped("The NVIDIA DLSS5 DLL and model files are separate local files. Choose a model source above, or configure weight_source in Phoenix.json.");
    };
    ui.performance_details = [] { draw_native_performance_details(); };
#if CNR_PHOENIX_DIAGNOSTIC
    ImGui::TextUnformatted("Paired DLL diagnostic | game controls locked");
    if (ImGui::Button("Save configuration")) ui.save();
    ImGui::TextWrapped("%s", ui.status.c_str());
    if (!ui.error.empty()) ImGui::TextWrapped("%s", ui.error.c_str());
    phoenix_ui::draw_timing(ui);
    draw_native_performance_details();
#else
    if (phoenix_ui::draw_controls(ui)) {
        addon_state.live.reset_requested = true;
        addon_state.timing_settings_dirty = true;
        save_live_controls(runtime);
    }
#endif
}

phoenix::direct::AppearanceTextures harvest_appearance(ID3D12GraphicsCommandList* cmd,
    const NVSDK_NGX_Parameter* params,FrameEvidence& evidence) {
    phoenix::direct::AppearanceTextures textures;
    unsigned profile=0;params->Get("Phoenix.Appearance.Contract",&profile);
    if(profile!=1) {
        evidence.experimental_appearance_contract=state().live.appearance;
        state().bridge_inputs["appearance"]={{"source",state().live.appearance?"phoenix_panel":"none"},
            {"reason","plane_producer_missing"},{"control_map",false},{"ui",false},{"distortion",false},
            {"generation",0u},{"vendor_validated",false}};
        return textures;
    }
    evidence.experimental_appearance_contract=true;
    const char* keys[]={"DLSSNR.ControlMask","DLSSNR.UI","DLSSNR.UIAlpha","DLSSNR.Backbuffer","DLSSNR.BidirectionalDistortionField"};
    ID3D12Resource** targets[]={&textures.mask,&textures.ui,&textures.alpha,&textures.backbuffer,&textures.field};
    for(unsigned i=0;i<5;++i) {
        *targets[i]=parameter_resource(params,keys[i]);if(!*targets[i])continue;
        D3D12_RESOURCE_STATES observed{};
        check(observation::epoch_state(cmd,*targets[i],observed)&&observation::current_provider_resource(cmd,*targets[i]),
              "Developer appearance resource needs a fresh observed state");
        evidence.optional_states[keys[i]]=observed;
    }
    unsigned generation=0;params->Get("Phoenix.Appearance.Generation",&generation);
    evidence.appearance_generation=generation;
    state().bridge_inputs["appearance"]={{"source",active_feeder_v2()?"feeder_v2":"engine"},
        {"profile","phoenix-experimental-rgb/1"},{"control_map",textures.mask!=nullptr},
        {"ui",textures.ui||textures.alpha},{"distortion",textures.field!=nullptr},
        {"generation",generation},{"vendor_validated",false}};
    return textures;
}
void draw_phoenix_compact(reshade::api::effect_runtime* runtime) {
    if (phoenix_compat::applies(runtime)) { phoenix_compat::compact(runtime); return; }
    auto& value = state();
    std::lock_guard<std::recursive_mutex> lock(value.mutex);
    load_live_controls(runtime);
    auto ui = make_panel_context(runtime, {}, value.weight_source.disk_model());
    ui.status = value.terminal ? "Stopped" : value.direct_runner || value.endpoint.session ? value.direct_status : "Waiting";
    if (phoenix_ui::draw_compact(ui)) {
        value.live.reset_requested = true;
        value.timing_settings_dirty = true;
        save_live_controls(runtime);
    }
}
void draw_comparison_hud(reshade::api::effect_runtime* runtime) {
    if (phoenix_compat::applies(runtime)) { phoenix_compat::comparison_hud(runtime); return; }
#if !CNR_PHOENIX_DIAGNOSTIC
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    const auto& presentation = addon_state.displayed;
    if (addon_state.terminal || !addon_state.display_tick ||
        GetTickCount64() - addon_state.display_tick > 2000)
        return;
    // Split views in both providers; direct fullscreen routes (ours, nvidia,
    // input) only get the single source label below. Native fullscreen
    // routes intentionally still draw nothing, as before.
    if (presentation.mode != phoenix::split &&
        !(addon_state.direct_game &&
          (presentation.mode == phoenix::ours || presentation.mode == phoenix::nvidia ||
           presentation.mode == phoenix::input)))
        return;
    ++addon_state.hud_draws;
    phoenix_ui::draw_presentation_hud(presentation);
#endif
}
} // namespace

extern "C" __declspec(dllexport) uint32_t WINAPI CnrPHOENIXFeederVersion() {
    return CNR_PHOENIX_DIAGNOSTIC ? 0u : PhoenixFeederAbiV1;
}
extern "C" __declspec(dllexport) HRESULT WINAPI CnrPHOENIXFeederEvaluateLocked(
    const PhoenixFeederFrameV1* frame, PhoenixFeederResultV1* result, uint64_t capture_token);
extern "C" __declspec(dllexport) uint32_t WINAPI CnrPHOENIXFeederVersionV2() {return CNR_PHOENIX_DIAGNOSTIC?0u:2u;}
extern "C" __declspec(dllexport) HRESULT WINAPI
CnrPHOENIXFeederEvaluate(const PhoenixFeederFrameV1* frame, PhoenixFeederResultV1* result) {
    if (!frame || !result || !PhoenixFeederContractV1(*frame, *result))
        return E_INVALIDARG;
    result->status = PhoenixFeederStatusV1::Deferred;
    result->recorded_frames = 0;
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    try {
        check(!CNR_PHOENIX_DIAGNOSTIC && (addon_state.direct_feeder || phoenix_present_stage) && addon_state.direct_game,
              "Direct feeder API requires input_provider=feeder_d3d12 in a live addon");
        check(!addon_state.terminal, "Phoenix DLSS5 direct feeder is terminal");
        check(gpu_gate(), "Phoenix DLSS5 direct feeder is not armed");
        const bool need_nr = addon_state.panel.mode == phoenix::nvidia ||
            (addon_state.panel.mode == phoenix::split &&
             (addon_state.panel.source_a == 1 || addon_state.panel.source_b == 1));
        if (frame->stream_id != 0 && addon_state.panel.uplift && need_nr) {
            addon_state.direct_status =
                "Stereo feeder supports Input/Phoenix DLSS5 sources only; NVIDIA DLSS5 comparison needs isolated NR eye features";
            return E_NOTIMPL;
        }
        auto* cmd = resolve_native_list(frame->command_list);
        check(resolve_queue(cmd) == frame->queue,
              "Phoenix DLSS5 feeder queue differs from the list's observed submitting queue");
        const auto output = frame->output->GetDesc(), motion = frame->motion->GetDesc();
        check(output.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
                  motion.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
                  output.SampleDesc.Count == 1 && motion.SampleDesc.Count == 1 &&
                  output.DepthOrArraySize == 1 && motion.DepthOrArraySize == 1 &&
                  output.MipLevels == 1 && motion.MipLevels == 1 &&
                  output.Width == motion.Width && output.Height == motion.Height &&
                  output.Format == DXGI_FORMAT_R16G16B16A16_FLOAT &&
                  phoenix::direct::motion_format(motion.Format),
              "Phoenix DLSS5 feeder requires native-size float color and motion textures");
        if (frame->depth) {
            const auto depth = frame->depth->GetDesc();
            check(depth.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
                      depth.SampleDesc.Count == 1 && depth.DepthOrArraySize == 1 &&
                      depth.MipLevels == 1 && depth.Width == output.Width &&
                      depth.Height == output.Height && depth.Format == DXGI_FORMAT_R32_FLOAT,
                  "Phoenix DLSS5 feeder depth requires native-size scalar R32_FLOAT");
        }
        ComPtr<ID3D12Device> device;
        phoenix_native::hr(cmd->GetDevice(IID_PPV_ARGS(&device)));
        ComPtr<IUnknown> device_identity;
        phoenix_native::hr(device.As(&device_identity));
        for (auto* resource : {frame->output, frame->motion, frame->depth}) {
            if (!resource) continue;
            ComPtr<ID3D12Device> owner;
            phoenix_native::hr(resource->GetDevice(IID_PPV_ARGS(&owner)));
            ComPtr<IUnknown> owner_identity;
            phoenix_native::hr(owner.As(&owner_identity));
            if (owner_identity != device_identity) {
                // ReShade's resource GetDevice can report its device proxy.
                // Compare canonical identities after the documented unwrap,
                // rather than treating that same-device proxy as a foreign GPU.
                ComPtr<IUnknown> unwrapped;
                if (SUCCEEDED(owner->QueryInterface(kUnwrappedObject,
                        reinterpret_cast<void**>(unwrapped.GetAddressOf()))) && unwrapped) {
                    ComPtr<ID3D12Device> original;
                    phoenix_native::hr(unwrapped.As(&original));
                    phoenix_native::hr(original.As(&owner_identity));
                }
            }
            check(owner_identity == device_identity, "Phoenix DLSS5 feeder texture belongs to another device");
            D3D12_RESOURCE_STATES observed{};
            check(observation::epoch_state(cmd, resource, observed) &&
                      observed == D3D12_RESOURCE_STATE_COMMON &&
                      observation::current_provider_resource(cmd, resource),
                  "Phoenix DLSS5 feeder texture requires fresh observed COMMON state");
        }
        ++addon_state.feeder_calls;
        if (addon_state.feeder_stream && *addon_state.feeder_stream != frame->stream_id &&
            addon_state.direct_runner && !addon_state.direct_runner->idle())
            return S_FALSE;
        if (addon_state.live.reset_requested || addon_state.last_mode != addon_state.panel.mode)
            addon_state.feeder_reset_pending.fill(true);
        if (!addon_state.feeder_stream || *addon_state.feeder_stream != frame->stream_id) {
            // Park only a retired executor; each eye retains its own history.
            if (addon_state.feeder_stream) {
                addon_state.feeder_last_blocks[*addon_state.feeder_stream] = addon_state.last_transformer_blocks;
                addon_state.feeder_parked[*addon_state.feeder_stream] = std::move(addon_state.direct_runner);
                if (addon_state.live_timing) addon_state.live_timing->budget(nullptr);
                addon_state.feeder_parked_timing[*addon_state.feeder_stream] = std::move(addon_state.live_timing);
            }
            addon_state.direct_runner = std::move(addon_state.feeder_parked[frame->stream_id]);
            addon_state.live_timing = std::move(addon_state.feeder_parked_timing[frame->stream_id]);
            if (addon_state.live_timing) addon_state.live_timing->budget(&addon_state.depth_budget);
            addon_state.last_transformer_blocks = addon_state.feeder_last_blocks[frame->stream_id];
            addon_state.timing_settings_dirty = true;
            addon_state.feeder_stream = frame->stream_id;
        }
        const bool reset_history = frame->reset || addon_state.feeder_reset_pending[frame->stream_id];
        addon_state.live.reset_requested = reset_history;
        // No fake feature is registered: the direct renderer needs only the
        // borrowed frame resources and controls, carried by this local adapter.
        const phoenix_addon::FeederParameters parameters(*frame);
        const auto before = addon_state.recorded;
        addon_state.feeder_applied_reset = false;
        addon_state.feeder_nr_initialization_recorded = false;
        direct_game_frame(cmd, nullptr, &parameters);
        result->recorded_frames = addon_state.recorded;
        if (addon_state.terminal) return E_FAIL;
        if (addon_state.panel.mode == phoenix::input || !addon_state.panel.uplift ||
            (addon_state.panel.processing.enabled && addon_state.panel.processing.hook == 0)) {
            result->status = PhoenixFeederStatusV1::Bypass;
            return S_OK;
        }
        if (addon_state.recorded > before) {
            addon_state.feeder_reset_pending[frame->stream_id] = false;
            ++addon_state.feeder_stream_recorded[frame->stream_id];
            if (addon_state.feeder_applied_reset) ++addon_state.feeder_stream_resets[frame->stream_id];
            result->status = PhoenixFeederStatusV1::Recorded;
            return S_OK;
        }
        if (addon_state.direct_runner && addon_state.direct_runner->pending(cmd)) {
            addon_state.direct_runner->invalidate();
            addon_state.terminal = true;
            error("Phoenix DLSS5 feeder refused after claiming its command list; abandon the recording");
            return E_FAIL;
        }
        if (addon_state.feeder_nr_initialization_recorded) {
            // CreateFeature recorded setup work but not an image. Submit it
            // safely as an unchanged frame, never discard it as Deferred and
            // later evaluate a feature whose initialization never reached GPU.
            result->status = PhoenixFeederStatusV1::Bypass;
            return S_OK;
        }
        return S_FALSE;
    } catch (const std::exception& exception) {
        addon_state.direct_status = exception.what();
        addon_state.live.reset_requested = true;
        return E_FAIL;
    } catch (...) {
        return E_FAIL;
    }
}

extern "C" __declspec(dllexport) HRESULT WINAPI CnrPHOENIXFeederEvaluateLocked(
    const PhoenixFeederFrameV1* frame, PhoenixFeederResultV1* result, uint64_t capture_token) {
    if (!capture_token) return E_INVALIDARG;
    phoenix_addon::phoenix::LockedCaptureScope scope(capture_token);
    return CnrPHOENIXFeederEvaluate(frame, result);
}

// Optional explicit engine registration. The environment-driven first-live
// path uses the built-in observed-state resolver instead.
extern "C" __declspec(dllexport) BOOL WINAPI
CnrPHOENIXRegister(const phoenix_addon::Endpoint* endpoint) {
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    try {
#if CNR_PHOENIX_DIAGNOSTIC
        check(
            false,
            "phoenix diagnostic requires sandbox configuration and its separate GPU gate; explicit registration disabled");
#endif
        check(endpoint && endpoint->session && endpoint->feature && endpoint->resolve,
              "phoenix incomplete endpoint");
        check(!addon_state.endpoint.session, "phoenix endpoint already registered");
        check(!addon_state.claimed_feature ||
                  *addon_state.claimed_feature == native_feature_key(endpoint->feature),
              "phoenix previously claimed feature not released");
        check(!addon_state.automatic, "phoenix automatic and explicit endpoints cannot be mixed");
        install_native(endpoint->session->queue());
        install_dispatch_audit();
        addon_state.endpoint = *endpoint;
        addon_state.claimed_feature = native_feature_key(endpoint->feature);
        addon_state.terminal = false;
        addon_state.error.clear();
        return TRUE;
    } catch (const std::exception& feature_entry) {
        error(feature_entry.what());
        return FALSE;
    } catch (...) {
        error("phoenix registration failed");
        return FALSE;
    }
}
extern "C" __declspec(dllexport) BOOL WINAPI CnrPHOENIXAbandoned(ID3D12GraphicsCommandList* cmd) {
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    try {
        check(bool(addon_state.endpoint.session), "phoenix no endpoint");
        addon_state.endpoint.session->abandoned(cmd);
        return TRUE;
    } catch (const std::exception& feature_entry) {
        error(feature_entry.what());
        return FALSE;
    } catch (...) {
        return FALSE;
    }
}

extern "C" __declspec(dllexport) BOOL WINAPI CnrPHOENIXUnregister() {
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    if (addon_state.endpoint.session && !addon_state.endpoint.session->can_destroy())
        return FALSE;
    retire();
    collect();
    // Unregister does not hand ownership back. Explicit panel routing is a
    // separate, reset-and-audited operation inside the still-owned boundary.
    addon_state.terminal = true;
    return TRUE;
}

// UI metadata is queried at most once a second by the helper's pipe worker.
// It is separate from the deliberately small per-frame completion snapshot.
namespace {
reshade::api::effect_runtime* backbuffer_owner = nullptr;
void observe_backbuffer_input(reshade::api::effect_runtime* runtime,
    ID3D12GraphicsCommandList* list, ID3D12CommandQueue* queue) {
    auto& value = state();
    std::lock_guard<std::recursive_mutex> lock(value.mutex);
    configure();
    check((value.direct_feeder || phoenix_present_stage) && value.direct_game && gpu_gate() && !value.terminal,
          "In-process observation requires an armed feeder_d3d12 configuration");
    check(!backbuffer_owner || backbuffer_owner == runtime,
          "Another window owns the in-process Phoenix DLSS5 session");
    observation::install(list, true);
    install_submission_observer(queue);
}
void prepare_backbuffer_input(reshade::api::effect_runtime* runtime,
                             const phoenix::Settings& settings) {
    auto& value = state();
    std::lock_guard<std::recursive_mutex> lock(value.mutex);
    configure();
    check((value.direct_feeder || phoenix_present_stage) && value.direct_game && gpu_gate() && !value.terminal,
          "In-process backbuffer processing requires an armed feeder_d3d12 configuration");
    check(!backbuffer_owner || backbuffer_owner == runtime,
          "Another window owns the in-process Phoenix DLSS5 session; use an isolated helper for this window");
    check(settings.valid() && settings.execution_device == "native",
          "Game Present requires Native RTX; use a final-frame frontend for portable processing");
    const bool model_changed = settings.model_source != value.active_model_source ||
        settings.nvidia_dll_folder != value.active_nvidia_dll_folder ||
        settings.model_folder != value.active_model_folder || (settings.model_dll != value.active_model_dll || settings.model_id != value.active_model_id);
    // The frame path validates and publishes the replacement transactionally.
    // Dropping the active bundle here would lose rollback on a bad selection.
    const bool reset = value.live.reset_requested ||
        phoenix_compat::helper_settings_changed(value.panel, settings);
    value.panel = settings;
    value.live.reset_requested = reset || model_changed;
    value.live_loaded = true;
    value.timing_settings_dirty = true;
    install_native(nullptr);
    backbuffer_owner = runtime;
}
void release_backbuffer_input(reshade::api::effect_runtime* runtime) {
    auto& value = state();
    std::lock_guard<std::recursive_mutex> lock(value.mutex);
    if (backbuffer_owner == runtime) {
        backbuffer_owner = nullptr;
        value.live.reset_requested = true;
    }
}
void update_live_tuning(const phoenix::Settings& settings) {
    auto& value = state();
    std::lock_guard<std::recursive_mutex> lock(value.mutex);
    check(settings.valid(), "Invalid live controls");
    if (phoenix::copy_live_tuning(value.panel, settings)) {
        value.live.reset_requested = true;
        value.timing_settings_dirty = true;
    }
}
void forget_backbuffer_commands(ID3D12GraphicsCommandList* cmd) noexcept {
    try {
        auto& value = state();
        std::lock_guard<std::recursive_mutex> lock(value.mutex);
        if (value.direct_runner && value.direct_runner->pending(cmd)) {
            value.direct_runner->abandon_unsubmitted(cmd);
            value.terminal = true;
        }
        if (value.live_timing) value.live_timing->abandoned(cmd);
        for (const auto& timer : value.retired_timing) timer->abandoned(cmd);
        value.submitted_lists.erase(cmd);
        value.states.destroyed_list(uint64_t(cmd));
        value.ownership.erase(cmd);
        observation::erase(cmd);
        for (auto iterator = value.originals.begin(); iterator != value.originals.end();)
            iterator = iterator->first == cmd || iterator->second == cmd ? value.originals.erase(iterator) : std::next(iterator);
    } catch (...) { state().terminal = true; }
}
} // namespace
extern "C" __declspec(dllexport) HRESULT WINAPI CnrPHOENIXFeederEvaluateV2(
    const PhoenixFeederFrameV2* frame,PhoenixFeederResultV1* result) {
    if(!frame||!result||!PhoenixFeederContractV2(*frame,*result)||active_feeder_v2())return E_INVALIDARG;
    struct Scope {~Scope(){active_feeder_v2()=nullptr;}} scope;
    active_feeder_v2()=frame;
    return CnrPHOENIXFeederEvaluate(&frame->frame,result);
}
extern "C" __declspec(dllexport) BOOL WINAPI CnrPHOENIXFeederLiveSettings(const char* text, uint32_t bytes) {
    if (!text || !bytes || bytes > 8192 || CNR_PHOENIX_DIAGNOSTIC) return FALSE;
    auto& value = state();
    std::lock_guard<std::recursive_mutex> lock(value.mutex);
    try {
        check(value.direct_feeder && !value.terminal, "Live tuning requires an active feeder");
        auto next = value.panel;
        phoenix_compat::apply_live_tuning_patch(next, Json::parse(text, text + bytes));
        update_live_tuning(next);
        return TRUE;
    } catch (...) { return FALSE; }
}
extern "C" __declspec(dllexport) uint32_t WINAPI CnrPHOENIXInProcessBackbufferVersion() {
    return CNR_PHOENIX_DIAGNOSTIC ? 0u : 1u;
}
extern "C" __declspec(dllexport) uint32_t WINAPI CnrPHOENIXNativeTlsV3() {
    return CNR_PHOENIX_DIAGNOSTIC ? 0u : 3u;
}
extern "C" __declspec(dllexport) uint32_t WINAPI CnrPHOENIXFeederUiJson(char* destination,
                                                                    uint32_t capacity) {
    auto& value = state();
    std::lock_guard<std::recursive_mutex> lock(value.mutex);
    try {
        auto info = bridge_snapshot();
        info["schema"] = "phoenix-panel/1";
        info["model_validation"]["key"] = value.model_validation.poll().key;
        info["runtime_root"] = phoenix_model::source_path_string(phoenix::portable::native_runtime_root());
        info["bundle_path"] = phoenix_model::source_path_string(value.bundle_path);
        info["overrides_path"] = phoenix_model::source_path_string(overrides_path());
        info["override_status"] = value.override_status;
        info["error"] = value.error;
        info["last_mode"] = value.last_mode;
        info["transformer_blocks"] = value.last_transformer_blocks;
        info["budget_estimate_ms"] = value.depth_budget.ema();
        info["history_resets"] = value.live_resets_recorded;
        info["markers"] = value.markers;
        if (value.direct_runner) {
            const auto& geometry = value.direct_runner->resolution();
            info["resolution"] = {{"width", geometry.width}, {"height", geometry.height},
                {"image_width", geometry.image_width}, {"image_height", geometry.image_height},
                {"output_width", geometry.output_width}, {"output_height", geometry.output_height}};
            info["upscaler_status"] = value.direct_runner->upscale_status();
        }
        info["gpu_history"] = Json::array();
        if (value.live_timing) {
            const auto& history = value.live_timing->history();
            const auto count = value.live_timing->history_count();
            const auto offset = value.live_timing->history_offset();
            for (size_t index = 0; index < count; ++index)
                info["gpu_history"].push_back(history[(offset + index) % history.size()]);
        }
        const auto text = info.dump();
        if (text.size() >= 1024u * 1024u) return 0;
        const auto required = uint32_t(text.size() + 1);
        if (destination && capacity >= required) std::memcpy(destination, text.c_str(), required);
        return required;
    } catch (...) { return 0; }
}

extern "C" __declspec(dllexport) BOOL WINAPI CnrPHOENIXFeederUiAction(uint32_t command,
                                                                  uint32_t argument) {
    if (!phoenix_ui::valid_action(command)) return FALSE;
    auto& value = state();
    std::lock_guard<std::recursive_mutex> lock(value.mutex);
    try {
        switch (phoenix_ui::Action(command)) {
        case phoenix_ui::Action::ResetHistory: value.live.reset_requested = true; break;
        case phoenix_ui::Action::CaptureInputs:
            if (argument > 64) return FALSE;
            arm_input_capture(argument ? argument : 8); break;
        case phoenix_ui::Action::ReloadOverrides: poll_overrides(nullptr, true); break;
        case phoenix_ui::Action::Marker:
            ++value.markers;
            record_event("phoenix_marker", {{"index", value.markers}, {"mode", value.panel.mode},
                                            {"frame", value.routed_frames}}); break;
        case phoenix_ui::Action::ClearTiming:
            if (value.live_timing) value.live_timing->restart();
            value.worker_timing.clear(); break;
        case phoenix_ui::Action::ValidateModel:
            value.model_validation.validate(value.weight_source, phoenix::portable::native_runtime_root()); break;
        }
        return TRUE;
    } catch (const std::exception& error) { value.error = error.what(); return FALSE; }
}

// A narrow, in-memory completion snapshot for the compatibility helper. The
// full diagnostic report remains available, but filesystem writes, GPU memory
// sampling and timing-history serialization do not belong in every frame call.
extern "C" __declspec(dllexport) uint32_t WINAPI CnrPHOENIXFeederProgressJson(char* destination,
                                                                          uint32_t capacity) {
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    try {
        uint64_t completed = addon_state.completed_retired;
        if (addon_state.direct_runner) completed += addon_state.direct_runner->completed();
        for (const auto& runner : addon_state.feeder_parked)
            if (runner) completed += runner->completed();
        if (addon_state.endpoint.session) completed += addon_state.endpoint.session->completed();
        for (const auto& session : addon_state.retired)
            completed += session->completed();
        const Json progress{
            {"schema", "production-game-phoenix/2"},
            {"input_provider", (addon_state.direct_feeder || phoenix_present_stage) ? "feeder_d3d12" : "game_dlss"},
            {"terminal", addon_state.terminal},
            {"completed_frames", completed},
            {"phoenix_routing", {{"frames", addon_state.routed_frames}, {"active_mode", addon_state.last_mode}}},
            {"direct_game", {{"enabled", addon_state.direct_game}, {"status", addon_state.direct_status}}},
            {"worker_execution", addon_state.direct_runner ? addon_state.direct_runner->execution_status()
                                                          : Json{{"kind", "not_started"}}}};
        const auto text = progress.dump();
        const auto required = uint32_t(text.size() + 1);
        if (destination && capacity >= required) std::memcpy(destination, text.c_str(), required);
        return required;
    } catch (...) { return 0; }
}

extern "C" __declspec(dllexport) BOOL WINAPI CnrPHOENIXSummary(const wchar_t* path) {
    auto& addon_state = state();
    std::lock_guard<std::recursive_mutex> lock(addon_state.mutex);
    try {
        auto launches = addon_state.candidate_launches;
        auto completed_frames = addon_state.completed_retired;
        if (addon_state.direct_runner) {
            launches += addon_state.direct_runner->launches();
            completed_frames += addon_state.direct_runner->completed();
        }
        for (const auto& runner : addon_state.feeder_parked)
            if (runner) {
                launches += runner->launches();
                completed_frames += runner->completed();
            }
        if (addon_state.endpoint.session)
            launches += addon_state.endpoint.session->candidate_launches();
        if (addon_state.endpoint.session)
            completed_frames += addon_state.endpoint.session->completed();
        for (const auto& session : addon_state.retired)
            launches += session->candidate_launches();
        for (const auto& session : addon_state.retired)
            completed_frames += session->completed();
        Json report{
            {"schema", "production-game-phoenix/2"},
            {"claimed_frames", addon_state.claimed},
            {"recorded_frames", addon_state.recorded},
            {"feeder_calls", addon_state.feeder_calls},
            {"feeder_stream_recorded", addon_state.feeder_stream_recorded},
            {"feeder_stream_resets", addon_state.feeder_stream_resets},
            {"input_provider", addon_state.direct_feeder ? "feeder_d3d12" : addon_state.direct_game ? "game_dlss" : "native_nr"},
            {"input_captures", addon_state.captures_taken},
            {"capture_base", addon_state.capture_base},
            {"submitted_frames", addon_state.submitted},
            {"rejected_frames", addon_state.rejected},
            {"unclaimed_original_evaluations", addon_state.unclaimed_native},
            {"completed_frames", completed_frames},
            {"bounded_run_gpu_complete", completed_frames == addon_state.frame_limit &&
                                             addon_state.submitted == addon_state.frame_limit &&
                                             !addon_state.terminal && addon_state.rejected == 0},
            {"blocked_submission_batches", addon_state.blocked_submissions},
            {"budget_refusals_after_bounded_run", addon_state.budget_refusals},
            {"bounded_run_recording_complete", addon_state.submitted == addon_state.frame_limit},
            {"candidate_original_evaluations", 0},
            {"candidate_native_inference_dispatches",
             addon_state.direct_game ? Json(nullptr)
                                     : Json(addon_state.dispatch_counts.other_attempted)},
            {"live_selected_native_evaluations", addon_state.direct_game
                                                     ? uint64_t(0)
                                                     : addon_state.mode_frames[phoenix::nvidia] +
                                                           addon_state.mode_frames[phoenix::split]},
            {"phoenix_routing",
             {{"frames", addon_state.routed_frames},
              {"switches", addon_state.switches},
              {"active_mode", addon_state.last_mode},
              {"test_period", addon_state.routing_test_period},
              {"mode_frames", addon_state.mode_frames},
              {"own_launches", addon_state.mode_own},
              {"native_launches", addon_state.mode_native},
              {"events", addon_state.route_events},
              {"scratch_leases", addon_state.route_outputs.size()},
              {"pairs",
               [&] {
                   Json pairs = Json::array();
                   for (const auto& route_pair : addon_state.route_pairs)
                       pairs.push_back(route_pair->result());
                   return pairs;
               }()}}},
            {"phoenix_comparison",
             {{"frames_by_mode", addon_state.comparison_frames},
              {"hud_draws", addon_state.hud_draws},
              {"resource_leases", addon_state.comparisons.size()},
              {"override_updates", addon_state.override_updates},
              {"override_status", addon_state.override_status}}},
            {"worker_timing", addon_state.worker_timing.report()},
            {"worker_input", addon_state.bridge_inputs},
            {"worker_execution", addon_state.direct_runner
                                     ? addon_state.direct_runner->execution_status()
                                     : Json{{"kind", "not_started"}}},
            {"phoenix_settings",
             {{"current", panel_json(addon_state.panel)},
              {"presets",
               [&] {
                   Json presets = Json::array();
                   for (const auto& preset : addon_state.presets)
                       presets.push_back(panel_json(preset));
                   return presets;
               }()},
              {"hotkeys", addon_state.hotkeys}}},
            {"native_dispatch_count_basis",
             "observed non-candidate LaunchCuKernelChain launches on claimed Evaluate thread; conservative native upper bound, not process-wide"},
            {"dispatch_observer_installed", addon_state.original_dispatch != nullptr},
            {"observed_own_cuda_launches", addon_state.dispatch_counts.own_attempted},
            {"observed_own_cuda_launches_accepted", addon_state.dispatch_counts.own_accepted},
            {"observed_non_candidate_cuda_launches", addon_state.dispatch_counts.other_attempted},
            {"observed_non_candidate_cuda_launches_accepted",
             addon_state.dispatch_counts.other_accepted},
            {"observed_own_cuda_calls", addon_state.dispatch_counts.own_calls},
            {"observed_non_candidate_cuda_calls", addon_state.dispatch_counts.other_calls},
            {"original_dll_loaded", GetModuleHandleW(L"nvngx_dlssnr.dll") != nullptr},
            {"feature_creation_dependency",
             "game may retain original DLL, feature, host allocations and weights"},
            {"automatic_resolver_enabled", addon_state.automatic},
            {"batched_dispatch", true},
            {"relax_barriers", false},
            {"minimal_barriers", false},
            {"skip_groups_mask", skip_groups_mask.load(std::memory_order_relaxed)},
            {"phoenix_live",
             {{"interactive", addon_state.interactive},
              {"settings_loaded", addon_state.live_loaded},
              {"override_strengths", addon_state.live.enabled},
              {"tone", addon_state.live.tone},
              {"structure", addon_state.live.structure},
              {"appearance_enabled",addon_state.live.appearance},
              {"intensity",addon_state.live.intensity},{"style",addon_state.live.style},
              {"reset_pending", addon_state.live.reset_requested},
              {"resets_recorded", addon_state.live_resets_recorded}}},
            {"in_place_game_directory", addon_state.in_place},
            {"native_dll_sha256", addon_state.native_dll_sha256},
            {"native_dll_is_reference_build", addon_state.native_dll_is_reference},
            {"sandbox_is_owned_copy", addon_state.in_place.empty()},
            // The profile is otherwise a list of bare indices. These are the
            // labels the classifier already uses, so a reader can see which
            // part of the network each timed group is.
            {"dispatch_groups",
             [&] {
                 static const char* labels[]{"stem",
                                             "encoder_32",
                                             "encoder_64",
                                             "encoder_128",
                                             "encoder_256",
                                             "encoder_split_512",
                                             "repack_2d_to_1d",
                                             "vit_1024",
                                             "repack_1d_to_2d",
                                             "decoder_split_512",
                                             "decoder_256",
                                             "decoder_128",
                                             "decoder_64",
                                             "decoder_32",
                                             "head"};
                 static const unsigned ends[]{1,  5,   9,   15,  23,  56,  57, 97,
                                              98, 131, 139, 145, 149, 153, 154};
                 Json out = Json::array();
                 unsigned first = 0;
                 for (unsigned iteration_index = 0; iteration_index < 15; ++iteration_index) {
                     out.push_back(
                         {{"group", iteration_index},
                          {"label", labels[iteration_index]},
                          {"first_launch", first},
                          {"launches", ends[iteration_index] - first},
                          {"chained", iteration_index < addon_state.group_batched.size() &&
                                          addon_state.group_batched[iteration_index] != 0}});
                     first = ends[iteration_index];
                 }
                 return out;
             }()},
            {"command_list_identity",
             "runtime proxy unwrapped through the documented interface and confirmed by independent object identity"},
            {"wrapped_evaluate_lists_unwrapped", addon_state.unwrapped_lists},
            {"queue_ownership_basis",
             "observed ExecuteCommandLists of this exact list; rechecked against the real submission before Execute"},
            {"claims_deferred_for_unproven_ownership", addon_state.ownership_deferrals},
            {"stage", addon_state.stage},
            {"stage_frame", addon_state.stage_frame},
            {"measurement_mode", addon_state.measure},
            {"measurement_claim_enabled", addon_state.claim_enabled},
            {"measurement_complete", addon_state.measurement_complete},
            {"measured_window_seconds", addon_state.measure_ms / 1000.0},
            {"measured_warmup_seconds", addon_state.warmup_ms / 1000.0},
            {"presents_total", addon_state.presents},
            {"presents_in_measured_window", addon_state.window_presents},
            {"measured_window_elapsed_ms",
             addon_state.window_start_tick
                 ? (addon_state.window_end_tick
                        ? addon_state.window_end_tick - addon_state.window_start_tick
                        : 0)
                 : 0},
            {"claimed_frames_in_measured_window",
             addon_state.claimed - addon_state.window_claimed_at_start},
            {"submitted_frames_in_measured_window",
             addon_state.submitted - addon_state.window_completed_at_start},
            {"gpu_timestamps",
             addon_state.timestamps ? addon_state.timestamps->report() : Json(nullptr)},
            {"phoenix_gpu_timing",
             addon_state.live_timing ? addon_state.live_timing->report() : Json(nullptr)},
            {"phoenix_depth_frames", addon_state.depth_frames},
            {"phoenix_active_depth", addon_state.last_transformer_blocks},
            {"phoenix_present_fps", addon_state.present_fps},
            {"process_id", uint64_t(GetCurrentProcessId())},
            {"swapchains_created", addon_state.swapchains_created},
            {"swapchains_destroyed", addon_state.swapchains_destroyed},
            {"swapchain_resizes", addon_state.swapchain_resizes},
            {"fullscreen_transitions", addon_state.fullscreen_transitions},
            {"last_backbuffer_width", addon_state.last_backbuffer_width},
            {"last_backbuffer_height", addon_state.last_backbuffer_height},
            {"last_backbuffer_format", addon_state.last_backbuffer_format},
            {"last_color_space", addon_state.last_color_space},
            {"last_fullscreen", addon_state.last_fullscreen},
            {"memory_samples", addon_state.memory_samples},
            {"working_set_first_bytes", addon_state.working_set_first},
            {"working_set_last_bytes", addon_state.working_set_last},
            {"working_set_peak_bytes", addon_state.working_set_peak},
            {"private_first_bytes", addon_state.private_first},
            {"private_last_bytes", addon_state.private_last},
            {"private_peak_bytes", addon_state.private_peak},
            {"adapter_local_first_bytes", addon_state.video_first},
            {"adapter_local_last_bytes", addon_state.video_last},
            {"adapter_local_peak_bytes", addon_state.video_peak},
            {"adapter_local_budget_bytes", addon_state.video_budget},
            {"lifecycle_events", addon_state.lifecycle_events},
            {"memory_series", addon_state.memory_series},
            {"series_interval_ms", addon_state.series_interval},
            {"owned_allocation_accounting", addon_state.owned_allocations.snapshot()},
            {"nvapi", phoenix_native::Api::lifetime_report()},
            {"host_bookkeeping",
             [&] {
                 auto& observed = observation::observer();
                 std::lock_guard<std::recursive_mutex> observer_lock(observed.mutex);
                 Json categories = Json::object(), sessions = Json::array();
                 uint64_t live = 0;
                 for (size_t iteration_index = 0; iteration_index < size_t(host_memory::Tag::count);
                      ++iteration_index) {
                     const auto& counter = host_memory::counters()[iteration_index];
                     const auto bytes = counter.live.load();
                     live += bytes;
                     categories[host_memory::names[iteration_index]] = {
                         {"allocated_bytes", counter.allocated.load()},
                         {"deallocated_bytes", counter.freed.load()},
                         {"live_bytes", bytes},
                         {"peak_bytes", counter.peak.load()},
                         {"allocations", counter.allocations.load()},
                         {"deallocations", counter.deallocations.load()}};
                 }
                 std::set<const Session*> seen;
                 auto add = [&](const auto& session) {
                     if (session && seen.insert(session.get()).second)
                         sessions.push_back(session->host_references());
                 };
                 add(addon_state.endpoint.session);
                 for (const auto& session : addon_state.retired)
                     add(session);
                 for (const auto& entry : addon_state.submitted_lists)
                     add(entry.second);
                 uint64_t binaries = 0;
                 for (const auto& binary : addon_state.candidates)
                     binaries += binary.bytes.capacity();
                 return Json{
                     {"schema", "phoenix-host-bookkeeping/1"},
                     {"counted_container_live_bytes", live},
                     {"categories", categories},
                     {"observer", observation::reference_snapshot()},
                     {"sessions", sessions},
                     {"submitted_session_reference_slots", addon_state.submitted_lists.size()},
                     {"unique_referenced_sessions", seen.size()},
                     {"shared_weight_capacity_bytes", 0},
                     {"shared_kernel_byte_capacity", binaries},
                     {"ledger_pending_transitions", addon_state.states.transition_entries()},
                     {"ledger_destroyed_pending_entries",
                      addon_state.states.destroyed_pending_entries()},
                     {"scope",
                      "Allocator-requested bytes for named containers, not heap allocator overhead or driver backing; pointer counts are reference slots, not COM refcounts"}};
             }()},
            {"claim_evaluations_bypassed", addon_state.claim_bypassed},
            {"claims_released_after_engine_moved_on", addon_state.claim_releases_by_abandonment},
            {"command_list_identity_failures", addon_state.identity_failures},
            {"last_command_list_identity_refusal", addon_state.identity_refusal},
            {"last_queue_ownership_refusal", addon_state.ownership_refusal},
            {"last_output_state_refusal", addon_state.output_state_refusal},
            {"execute_indirect", observation::indirect_summary()},
            {"committed_resource_observation", observation::committed_summary()},
            {"device_identity", addon_state.device_identity},
            {"observed_command_lists", uint64_t(observation::observer().commands.size())},
            {"observed_submission_ownership_entries", uint64_t(addon_state.ownership.size())},
            {"engine_output_state_basis", addon_state.output_state_basis},
            {"cross_thread_submission_observed",
             addon_state.endpoint.session ? addon_state.endpoint.session->cross_thread_submission()
                                          : false},
            {"ledger_tracked_resources", addon_state.states.tracked_resources()},
            {"ledger_open_recordings", addon_state.states.open_recordings()},
            {"ledger_forgotten", addon_state.states.forgotten()},
            {"sampling_source",
             "native stem QWORD constructor/reset=0; copy at 0x60f40 then increment at 0x60f44"},
            {"last_sampling_index", addon_state.last_sampling},
            {"own_candidate_launches_attempted", launches},
            {"generations", addon_state.generations},
            {"feature_releases", addon_state.releases},
            {"abandoned_recordings", addon_state.resets},
            {"feature_creations_observed", addon_state.observed_creations},
            {"native_hook_installed", addon_state.original_evaluate != nullptr},
            {"ngx_export_hooks", addon_state.ngx_export_hooks},
            {"ngx_features_seen", addon_state.ngx_features_seen},
            {"sl_export_hooks", addon_state.sl_export_hooks},
            {"sl_tags_seen", addon_state.sl_tags_seen},
            {"sl_evaluates_seen", addon_state.sl_evaluates_seen},
            {"sl_template",
             [&] {
                 Json template_status{{"present", false}};
                 const auto* tpl = addon_state.sl_observed.drive_template();
                 if (!tpl)
                     return template_status;
                 template_status["present"] = true;
                 template_status["feature"] = tpl->feature;
                 template_status["color_w"] = tpl->color_w;
                 template_status["color_h"] = tpl->color_h;
                 template_status["color_format"] = tpl->color_format;
                 template_status["motion"] = tpl->motion;
                 template_status["depth"] = tpl->depth;
                 template_status["out_w"] = tpl->out_w;
                 template_status["out_h"] = tpl->out_h;
                 template_status["viewports"] = tpl->viewports;
                 template_status["evaluates"] = tpl->evaluates;
                 return template_status;
             }()},
            {"sl_last_tag_frame", addon_state.sl_observed.last_tag_frame()},
            {"sl_global_tags", addon_state.sl_observed.global_tags()},
            {"direct_game",
             {{"enabled", addon_state.direct_game},
              {"calls", addon_state.direct_calls},
              {"deferred", addon_state.direct_deferred},
              {"busy", addon_state.direct_busy},
              {"status", addon_state.direct_status},
              {"native_nr_required", false},
              {"exposure_flips",
               addon_state.direct_runner ? addon_state.direct_runner->exposure_flips() : 0},
              {"exposure_max_delta_pct", addon_state.direct_runner
                                             ? addon_state.direct_runner->exposure_max_delta_pct()
                                             : 0}}},
            {"selfdrive",
             {{"renodx_present", bool(GetModuleHandleW(L"renodx-dlss5.addon64"))},
              {"hosted", addon_state.selfdrive.handle != nullptr && !addon_state.selfdrive.failed},
              {"template_valid",
               sl_shim::negotiate_sl_sizes(addon_state.sl_observed.drive_template()).valid},
              {"created_id", addon_state.selfdrive.feature_id},
              {"retirements", addon_state.nvidia_retirements},
              {"evaluates", addon_state.selfdrive.evaluates},
              {"depth_present", addon_state.selfdrive.depth_present},
              {"depth_proven", addon_state.selfdrive.depth_proven},
              {"depth_typed", addon_state.selfdrive.depth_typed},
              {"failed", addon_state.selfdrive.failed},
              {"warmup", addon_state.selfdrive.warmup},
              {"dll_bytes", addon_state.selfdrive.dll_bytes},
              {"last_create_result", addon_state.selfdrive.last_create_result},
              {"last_evaluate_result", addon_state.selfdrive.last_evaluate_result},
              {"abandons", addon_state.selfdrive.abandons},
              {"nr_status", addon_state.direct_nr_status}}},
            {"sl_last_table",
             [&] {
                 Json report_rows = Json::array();
                 for (const auto& tag_row : addon_state.sl_observed.last_table()) {
                     report_rows.push_back({{"type", tag_row.type},
                                            {"w", tag_row.w},
                                            {"h", tag_row.h},
                                            {"format", tag_row.format},
                                            {"ex_w", tag_row.ex_w},
                                            {"ex_h", tag_row.ex_h}});
                 }
                 return report_rows;
             }()},
            {"sl_features",
             [&] {
                 Json report_rows = Json::array();
                 for (const auto& feature_entry : addon_state.sl_observed.features()) {
                     const auto& feature_inputs = feature_entry.second;
                     report_rows.push_back({{"feature", feature_inputs.feature},
                                            {"evaluates", feature_inputs.evaluates},
                                            {"last_frame", feature_inputs.last_frame},
                                            {"color_w", feature_inputs.color_w},
                                            {"color_h", feature_inputs.color_h},
                                            {"motion", feature_inputs.motion},
                                            {"depth", feature_inputs.depth},
                                            {"tags", feature_inputs.tags},
                                            {"viewports", feature_inputs.viewports}});
                 }
                 return report_rows;
             }()},
            {"retired_generations", addon_state.retired.size()},
            // Preserve the old report field for existing external test controllers.
            {"parent_gpu_gate_present", gpu_gate()},
            {"gpu_recording_allowed", gpu_gate()},
            {"requires_gpu_ready_file", addon_state.require_gpu_ready_file},
            {"gpu_ready_file", addon_state.require_gpu_ready_file
                                   ? Json(phoenix_model::source_path_string(addon_state.gate))
                                   : Json(nullptr)},
            {"terminal", addon_state.terminal},
            {"frames", addon_state.frames},
            {"automatic_game_contract_complete", false},
            {"live_gpu_parity_validated", false},
            {"error", addon_state.error}};
#if CNR_PHOENIX_DIAGNOSTIC
        report["schema"] = "production-game-phoenix-diagnostic/1";
        report["compiled_mode"] = "diagnostic_native_and_candidate";
        report.erase("candidate_original_evaluations");
        report.erase("candidate_native_inference_dispatches");
        report.erase("native_dispatch_count_basis");
        report["diagnostic_original_evaluations_attempted"] =
            addon_state.diagnostic_native_attempted;
        report["diagnostic_original_evaluations_succeeded"] =
            addon_state.diagnostic_native_succeeded;
        report["native_dispatch_count_basis"] =
            "non-candidate CUDA launch count on claimed Evaluate thread includes deliberate native Evaluate; not process-wide";
        report["production_candidate_only"] = false;
        report["benchmark_eligible"] = false;
        report["comparison_failed"] = addon_state.comparison_failed;
        report["capture_directory"] = phoenix_model::source_path_string(addon_state.capture_directory);
        report["pairs"] = Json::array();
        size_t completed = 0, exact = 0;
        for (const auto& pair : addon_state.pairs) {
            report["pairs"].push_back(pair->result());
            completed += pair->completed();
            exact += pair->completed() && pair->result().at("comparison") == "byte_exact";
        }
        report["completed_pairs"] = completed;
        report["byte_exact_pairs"] = exact;
        report["raw_capture_budget"] = addon_state.capture_limit;
        report["bounded_output_byte_exact"] = completed == addon_state.frame_limit &&
                                              exact == completed && !addon_state.terminal &&
                                              addon_state.rejected == 0;
        report["promotion_automatic"] = false;
#else
        report["compiled_mode"] = "production_candidate_only";
#endif
        // The summary is rewritten repeatedly while the run advances, and the
        // parent kills the process as soon as the budget is spent. Truncating
        // in place loses the whole file if the kill lands mid-write, which it
        // did once. Write beside it and rename, so a reader always sees a
        // complete summary: either the previous one or the new one.
        const std::filesystem::path final_path(path);
        auto temporary = final_path;
        temporary += L".writing";
        {
            std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
            check(bool(stream), "phoenix summary open failed");
            stream << report.dump(2) << '\n';
            stream.flush();
            if (!stream.good())
                return FALSE;
        }
        std::error_code failure;
        std::filesystem::rename(temporary, final_path, failure);
        if (failure) {
            std::filesystem::remove(temporary, failure);
            return FALSE;
        }
        return TRUE;
    } catch (...) {
        return FALSE;
    }
}

// ReShade may unload and reload an add-on inside one process. Under
// OptiScaler it did exactly that (`Unloading add-on ...` then
// `Add-on ... was not unregistered!` 0 ms later, then reloaded as
// "externally registered"), because FreeLibrary did not drop the module
// refcount to zero and DLL_PROCESS_DETACH never ran. Registration is
// therefore driven by ReShade's own AddonInit/AddonUninit, which it calls
// regardless of refcount, with DllMain kept only as a fallback for a host
// that does not call them. Both paths funnel through one idempotent pair so
// neither can register or unregister twice.
namespace {
std::atomic<bool> addon_registered{false};
HMODULE addon_module = nullptr;

bool addon_attach(HMODULE module) {
    bool expected = false;
    if (!addon_registered.compare_exchange_strong(expected, true))
        return true;
    if (!reshade::register_addon(module)) {
        addon_registered.store(false);
        return false;
    }
    addon_module = module;
#define _PHOENIX_STR_(horizontal_position) #horizontal_position
#define PHOENIX_STR(horizontal_position) _PHOENIX_STR_(horizontal_position)
#ifndef PHOENIX_BUILD_TAG
#define PHOENIX_BUILD_TAG "unstamped"
#endif
    reshade::log::message(reshade::log::level::info, "[Phoenix DLSS5] addon build " PHOENIX_STR(
                                                         PHOENIX_BUILD_TAG) " attached");
    state().self = module;
#if !CNR_PHOENIX_DIAGNOSTIC
    phoenix_compat::in_process_api = {prepare_backbuffer_input, release_backbuffer_input, observe_backbuffer_input,
        forget_backbuffer_commands, CnrPHOENIXFeederEvaluate, CnrPHOENIXFeederProgressJson, CnrPHOENIXFeederUiJson,
        CnrPHOENIXFeederUiAction, update_live_tuning};
    phoenix_compat::in_process_api.present_guides = [](ID3D12GraphicsCommandList* cmd,
        ID3D12CommandQueue* queue, PhoenixFeederFrameV1& frame) {
        auto& value=state();
        return value.present_guides && value.present_guides->acquire(cmd,queue,value.present_serial,frame);
    };
#endif
    phoenix_compat::attach(module);
    phoenix::direct::phoenix_crumb = phoenix_crumb;
    // Earliest install wins the inner hook position (see install_ngx_exports):
    // ReShade loads us before renodx, so hooking whatever NGX copies are
    // already present puts our trampolines under its Detours hooks, immune
    // to its async re-arming. Later-loading copies are picked up by the
    // present-driven retries. Never throws: a failed load path must not
    // refuse the addon.
    try {
        install_ngx_exports();
    } catch (...) {
    }
    try {
        install_sl_hooks();
    } catch (...) {
    }
    phoenix_native::record_progress = progress;
    observation::observer().reset_callback = command_reset;
    observation::observer().close_callback = command_close;
    // This host feeds destruction through destroy_resource below, so the
    // committed registry may keep a resource even when its own Release
    // cannot be hooked. The ASI host has no such event and stays strict.
    observation::observer().external_destruction_feed = true;
    reshade::register_event<reshade::addon_event::init_resource>(init_resource);
    reshade::register_event<reshade::addon_event::destroy_resource>(destroy_resource);
    reshade::register_event<reshade::addon_event::barrier>(barrier);
    reshade::register_event<reshade::addon_event::init_command_list>(init_command);
    reshade::register_event<reshade::addon_event::destroy_command_list>(destroy_command);
    reshade::register_event<reshade::addon_event::init_command_queue>(init_queue);
    reshade::register_event<reshade::addon_event::destroy_command_queue>(destroy_queue);
    reshade::register_event<reshade::addon_event::init_swapchain>(init_swapchain);
    reshade::register_event<reshade::addon_event::destroy_swapchain>(destroy_swapchain);
    reshade::register_event<reshade::addon_event::set_fullscreen_state>(set_fullscreen_state);
    reshade::register_event<reshade::addon_event::present>(present);
    reshade::register_event<reshade::addon_event::init_effect_runtime>(live_runtime_tick);
    reshade::register_event<reshade::addon_event::destroy_effect_runtime>(destroy_present_runtime);
    reshade::register_event<reshade::addon_event::reshade_present>(live_runtime_tick);
    reshade::register_event<reshade::addon_event::reshade_screenshot>(screenshot_saved);
    reshade::register_event<reshade::addon_event::reshade_overlay>(draw_comparison_hud);
    reshade::register_overlay("Phoenix DLSS5", draw_phoenix_overlay);
    reshade::register_overlay(nullptr, draw_phoenix_compact);
    return true;
}

void addon_detach(HMODULE module) {
    bool expected = true;
    if (!addon_registered.compare_exchange_strong(expected, false))
        return;
    phoenix_compat::detach();
    phoenix_compat::in_process_api = {};
    menu_helper::detach();
    // An active sidecar is pinned; process teardown releases the driver.
    reshade::unregister_overlay("Phoenix DLSS5", draw_phoenix_overlay);
    reshade::unregister_overlay(nullptr, draw_phoenix_compact);
    reshade::unregister_event<reshade::addon_event::reshade_present>(live_runtime_tick);
    reshade::unregister_event<reshade::addon_event::reshade_screenshot>(screenshot_saved);
    reshade::unregister_event<reshade::addon_event::reshade_overlay>(draw_comparison_hud);
    reshade::unregister_event<reshade::addon_event::init_effect_runtime>(live_runtime_tick);
    reshade::unregister_event<reshade::addon_event::destroy_effect_runtime>(destroy_present_runtime);
    reshade::unregister_event<reshade::addon_event::present>(present);
    reshade::unregister_event<reshade::addon_event::set_fullscreen_state>(set_fullscreen_state);
    reshade::unregister_event<reshade::addon_event::destroy_swapchain>(destroy_swapchain);
    reshade::unregister_event<reshade::addon_event::init_swapchain>(init_swapchain);
    reshade::unregister_event<reshade::addon_event::destroy_command_queue>(destroy_queue);
    reshade::unregister_event<reshade::addon_event::init_command_queue>(init_queue);
    reshade::unregister_event<reshade::addon_event::destroy_command_list>(destroy_command);
    reshade::unregister_event<reshade::addon_event::init_command_list>(init_command);
    reshade::unregister_event<reshade::addon_event::barrier>(barrier);
    reshade::unregister_event<reshade::addon_event::destroy_resource>(destroy_resource);
    reshade::unregister_event<reshade::addon_event::init_resource>(init_resource);
    reshade::unregister_addon(module ? module : addon_module);
    addon_module = nullptr;
}
} // namespace

// ReShade calls these before LoadLibrary's refcount matters and before
// FreeLibrary, so they run even when DLL_PROCESS_DETACH does not.
extern "C" __declspec(dllexport) bool AddonInit(HMODULE addon, HMODULE) {
    return addon_attach(addon);
}
extern "C" __declspec(dllexport) void AddonUninit(HMODULE addon, HMODULE) {
    addon_detach(addon);
}

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID) {
    // Fallback only. On a host that calls AddonInit this attach is a no-op
    // because ReShade has already registered us, and vice versa.
    if (reason == DLL_PROCESS_ATTACH) {
        if (!addon_attach(module))
            return FALSE;
    } else if (reason == DLL_PROCESS_DETACH)
        addon_detach(module);
    return TRUE;
}
