#include "../../src/native/support/policy.h"
#include "../../src/native/support/controls.h"
#include "../../src/native/support/ngx_shim.h"
#include "../../src/native/support/sl_shim.h"
#include "../../src/native/support/live_controls.h"
#include "../../src/native/support/routing.h"
#include "../../src/native/support/panel_visibility.h"
#include "../../src/native/support/overrides.h"
#include "../../src/native/support/kernel_resolution.h"
#include <map>
#include "../../src/native/support/ledger.h"
#include "../../src/native/support/trace_manifest.h"
#include "../../src/native/support/dispatch_log.h"
#include "../../src/native/support/panel_model.h"
#include <iostream>
#include <functional>
#include <limits>
using namespace phoenix_addon;
static unsigned cases = 0;
static void kernel_resolution_tests() {
    const std::string small = "small", large = "large";
    constexpr uint64_t threshold = uint64_t(5760) * 3240;
    const auto select = [&](unsigned width, unsigned height) {
        return phoenix_kernel::entry_for_resolution(small, large, threshold, width, height);
    };
    for (const auto extent : {std::pair<unsigned, unsigned>{1280, 720}, {1920, 1080},
                              {3840, 2160}, {5760, 3239}}) {
        check(select(extent.first, extent.second) == small, "small-input kernel preserved");
        ++cases;
    }
    for (const auto extent : {std::pair<unsigned, unsigned>{5760, 3240}, {3240, 5760},
                              {5761, 3241}, {7680, 4320}, {16384, 16384}}) {
        check(select(extent.first, extent.second) == large, "large-input kernel selected");
        ++cases;
    }
    check(phoenix_kernel::entry_for_resolution(small, "", 0, 7680, 4320) == small,
          "legacy kernel uses its original entry");
    ++cases;
    check(phoenix_kernel::entry_for_resolution(small, large, uint64_t(16384) * 16384,
                                              65536, 65536) == large,
          "pixel multiplication uses 64 bits");
    ++cases;
    for (const auto invalid : {std::pair<std::string, uint64_t>{large, 0},
                                {"", threshold}, {small, threshold},
                                {large, uint64_t(16384) * 16384 + 1}}) {
        check(!phoenix_kernel::valid_resolution_variant(small, invalid.first, invalid.second),
              "incomplete/invalid resolution variant rejected");
        ++cases;
    }
    check(phoenix_kernel::valid_resolution_variant(small, large, threshold) &&
              phoenix_kernel::valid_resolution_variant(small, "", 0),
          "valid tuned and legacy kernel policies");
    ++cases;
}
static void rejected(const std::function<void()>& action) {
    bool caught = false;
    try {
        action();
    } catch (const std::runtime_error&) {
        caught = true;
    }
    check(caught, "negative case was accepted");
    ++cases;
}
// Read a typed value from the packed control buffer without alignment assumptions.
template <class T, size_t N>
T read_packed_value(const std::array<uint8_t, N>& bytes, size_t offset) {
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}
struct Parameters {
    template <class T> int Get(const char*, T* value) const {
        *value = T(88);
        return -1;
    }
};
static void controls_tests() {
    const uint64_t handle = 1234, alias = 1234;
    check(native_feature_key(&handle) == native_feature_key(&alias),
          "native handle identity is key, not object address");
    ++cases;
    rejected([&] { native_feature_key(nullptr); });
    const auto controls = read_controls(Parameters{}, [](int result) { return result == 0; });
    check(controls.tone == 1 && controls.structure == 1 && controls.motion_x == 1 &&
              controls.motion_y == 1 && controls.skin == -1 && controls.style == 0 &&
              controls.reset == 0 && controls.enabled == 1 && controls.inverted == 1 &&
              controls.color.width == 0,
          "native defaults on failed getter");
    ++cases;
    rejected([&] { effective_controls(controls, true, false); });
    check(effective_controls(controls, true, true).reset == 1 && controls.reset == 0,
          "observed creation supplies local first reset without changing game parameters");
    ++cases;
    check(effective_controls(controls, false, true).reset == 0,
          "creation freshness applies only to first evaluation");
    ++cases;
    check(effective_controls(controls, true, false, true).reset == 1 && controls.reset == 0,
          "cold restart admits rebuild with local reset without changing game parameters");
    ++cases;
    Sampling sampling;
    check(sampling.begin(false) == 0, "constructor counter");
    ++cases;
    rejected([&] { sampling.begin(false); });
    sampling.commit();
    check(sampling.begin(false) == 1, "counter increment");
    ++cases;
    sampling.commit();
    check(sampling.begin(true) == 0, "counter reset");
    ++cases;
    sampling.commit();
    check(sampling.next() == 1, "post-reset increment");
    ++cases;
    sampling.invalidate();
    rejected([&] { sampling.begin(false); });
    rejected([&] { sampling.commit(); });
    const auto first = resolve_controls(controls, 640, 360, true, false, 0);
    check(!first.recurrent && first.history == Rectangle{} && first.motion == Rectangle{},
          "first-frame rectangles");
    ++cases;
    auto next = resolve_controls(controls, 640, 360, true, true, 7);
    check(next.recurrent && next.motion == next.color && next.history == next.color &&
              next.motion_x == 1.0f / 640 && next.motion_y == 1.0f / 360,
          "recurrent rectangles/scales");
    ++cases;
    {
        PhoenixLiveControls guides;
        guides.advanced = true;
        Controls host;
        host.motion_x = 1920.0f;
        host.motion_y = 1080.0f;
        const auto scaled = guides.apply(host);
        check(scaled.motion_x == 1920.0f && scaled.motion_y == 1080.0f,
              "default motion guides follow the host scale");
        ++cases;
    }
    {
        PhoenixLiveControls guides;
        guides.advanced = true;
        guides.motion_x = 2.0f;
        guides.motion_y = 0.5f;
        Controls host;
        host.motion_x = 1920.0f;
        host.motion_y = 1080.0f;
        const auto scaled = guides.apply(host);
        check(scaled.motion_x == 3840.0f && scaled.motion_y == 540.0f,
              "motion overrides multiply the host scale");
        ++cases;
    }
    {
        PhoenixLiveControls guides;
        Controls host;
        host.motion_x = 1920.0f;
        host.motion_y = 1080.0f;
        const auto scaled = guides.apply(host);
        check(scaled.motion_x == 1920.0f && scaled.motion_y == 1080.0f,
              "motion guides pass through without advanced override");
        ++cases;
    }
    auto absent = resolve_controls(controls, 640, 360, false, true, 8);
    check(!absent.recurrent && absent.sampling == 8 && absent.motion == Rectangle{},
          "missing motion preserves sampling");
    ++cases;
    auto changed = controls;
    changed.motion_x = -640;
    changed.motion_y = 720;
    changed.tone = 0.25f;
    changed.structure = 0.75f;
    changed.auto_mask = 1;
    changed.reset = -1;
    changed.inverted = 0;
    const auto reset = resolve_controls(changed, 640, 360, true, true, 0);
    check(!reset.recurrent && reset.motion_x == -1 && reset.motion_y == 2 && reset.skin == 0.75f &&
              reset.skin_structure == 0.75f && reset.inverted == 0,
          "reset/scales/auto-mask fallback");
    ++cases;
    changed.skin = 0.125f;
    check(resolve_controls(changed, 640, 360, true, true, 0).skin == 0.125f,
          "explicit skin override");
    ++cases;
    changed.auto_mask = 0;
    check(resolve_controls(changed, 640, 360, true, true, 0).skin == -1, "auto-mask off sentinel");
    ++cases;
    std::array<uint8_t, 264> stem;
    stem.fill(0xa5);
    std::array<uint8_t, 184> head;
    head.fill(0xa5);
    patch_stem(stem.data(), stem.size(), reset);
    patch_head(head.data(), head.size(), reset);
    check(read_packed_value<uint64_t>(stem, 8) == 0 && read_packed_value<uint64_t>(stem, 16) == 0 &&
              read_packed_value<uint64_t>(head, 88) == 0 &&
              read_packed_value<uint64_t>(head, 96) == 0 &&
              read_packed_value<uint64_t>(stem, 200) == 0,
          "reset descriptors/counter");
    ++cases;
    check(read_packed_value<float>(stem, 172) == 0.25f &&
              read_packed_value<float>(stem, 176) == 0.75f &&
              read_packed_value<float>(stem, 184) == 0.75f &&
              read_packed_value<float>(stem, 160) == -1 &&
              read_packed_value<float>(head, 168) == 2 &&
              read_packed_value<float>(stem, 196) == 0.0625f,
          "native scalar offsets");
    ++cases;
    check(read_packed_value<uint32_t>(stem, 192) == 1 &&
              read_packed_value<uint32_t>(head, 112) == 1 &&
              read_packed_value<Rectangle>(stem, 40) == Rectangle{} &&
              read_packed_value<Rectangle>(head, 116) == Rectangle{},
          "native flags/rectangles");
    ++cases;
    check(stem[208] == 0xa5 && head[172] == 0xa5 && stem[0] == 0xa5 && head[48] == 0xa5,
          "static geometry/pointers/gain preserved");
    ++cases;
    next.sampling = 0x1234567887654321ULL;
    patch_stem(stem.data(), stem.size(), next);
    check(read_packed_value<uint64_t>(stem, 200) == next.sampling, "complete 64-bit counter patch");
    ++cases;
    rejected([&] { patch_stem(stem.data(), 263, next); });
    rejected([&] { patch_head(head.data(), 183, next); });
    auto bad = controls;
    bad.color.x = 1;
    rejected([&] { resolve_controls(bad, 640, 360, true, true, 1); });
    bad = controls;
    bad.motion.width = 100;
    rejected([&] { resolve_controls(bad, 640, 360, true, true, 1); });
    bad = controls;
    bad.motion_x = std::numeric_limits<float>::quiet_NaN();
    rejected([&] { resolve_controls(bad, 640, 360, true, true, 1); });
    bad = controls;
    bad.intensity = 0.5f;
    rejected([&] { resolve_controls(bad, 640, 360, true, true, 1); });
    bad = controls;
    bad.style = 1;
    rejected([&] { resolve_controls(bad, 640, 360, true, true, 1); });
    bad = controls;
    bad.enabled = 0;
    rejected([&] { resolve_controls(bad, 640, 360, true, true, 1); });
    bad = controls;
    bad.ui_correction = 1;
    rejected([&] { resolve_controls(bad, 640, 360, true, true, 1); });
    // RE9 menu triple (measured live 2026-10-05): refused as a whole, and
    // the HDR override target (intensity 1, style 0, UI correction off)
    // resolves -- the override crosses exactly this boundary, nothing else.
    bad = controls;
    bad.intensity = 2.0f;
    bad.style = 2;
    bad.ui_correction = 1;
    rejected([&] { resolve_controls(bad, 640, 360, true, true, 1); });
    auto over = controls;
    over.intensity = 1.0f;
    over.style = 0;
    over.ui_correction = 0;
    check(resolve_controls(over, 640, 360, true, true, 1).sampling == 1,
          "hdr override triple resolves");
    ++cases;
    // The helper flips exactly the refused triple and nothing else; off
    // passes through untouched (enabled is never overridden).
    {
        auto menu = controls;
        menu.intensity = 2.0f;
        menu.style = 2;
        menu.ui_correction = 1;
        menu.tone = 0.5f;
        menu.enabled = 1;
        const auto hdr_controls = hdr_overridden_controls(menu, true);
        check(hdr_controls.intensity == 1.0f && hdr_controls.style == 0 &&
                  hdr_controls.ui_correction == 0,
              "hdr override flips triple");
        ++cases;
        check(hdr_controls.tone == 0.5f && hdr_controls.enabled == 1,
              "hdr override keeps everything else");
        ++cases;
        const auto off = hdr_overridden_controls(menu, false);
        check(off.intensity == 2.0f && off.style == 2 && off.ui_correction == 1,
              "hdr override off passes through");
        ++cases;
    }
}
static void ngx_shim_tests() {
    using namespace ngx_shim;
    for (const unsigned id : {0u, 1u, 13u}) {
        check(is_direct_game_feature(id), "direct game accepts legacy, SDK SR, and RR NGX IDs");
        ++cases;
    }
    for (const unsigned id : {2u, 11u, 18u, 1001u, ~0u}) {
        check(!is_direct_game_feature(id), "direct game rejects unrelated NGX and Streamline IDs");
        ++cases;
    }
    Observer observer;
    check(observer.size() == 0 && observer.find(1) == nullptr &&
              observer.drive_template() == nullptr,
          "empty observer");
    ++cases;
    observer.note_evaluate(0);
    observer.note_create(0, 13, 1, 1, 1, 1, "game");
    check(observer.size() == 0, "null keys ignored");
    ++cases;
    observer.note_create(7, 13, 1920, 1080, 1920, 1080, "re9.exe", 0xb);
    check(observer.find(7)->create_flags == 0xb,
          "retain game creation flags for NR depth convention");
    ++cases;
    check(observer.size() == 1, "create recorded");
    ++cases;
    check(observer.find(7) && observer.find(7)->evaluates == 0 &&
              observer.drive_template() == nullptr,
          "unevaluated feature is no template");
    ++cases;
    observer.note_evaluate(11);
    check(observer.find(11) && observer.find(11)->evaluates == 1 &&
              observer.find(11)->feature_id == 0,
          "untracked evaluates counted, never typed");
    ++cases;
    observer.note_evaluate(7);
    observer.note_evaluate(7);
    const Observation* tpl = observer.drive_template();
    check(tpl && tpl->key == 7 && tpl->evaluates == 2 && tpl->creator == "re9.exe",
          "evaluated RR is the template");
    ++cases;
    observer.note_create(9, 1, 1280, 720, 1280, 720, "re9.exe");
    observer.note_evaluate(9);
    check(observer.drive_template() && observer.drive_template()->key == 7,
          "non-RR never templates");
    ++cases;
    const auto sizes = negotiate_self_drive(tpl);
    check(sizes.valid && sizes.in_w == 1920 && sizes.in_h == 1080 && sizes.out_w == 1920 &&
              sizes.out_h == 1080,
          "self-drive mirrors template");
    ++cases;
    check(!negotiate_self_drive(nullptr).valid, "no template refuses");
    ++cases;
    Observation partial;
    partial.in_width = 640;
    check(!negotiate_self_drive(&partial).valid, "partial sizes refuse");
    ++cases;
    observer.note_release(7);
    check(observer.find(7) == nullptr && observer.drive_template() == nullptr, "release drops");
    ++cases;
    observer.note_release(99);
    check(observer.size() == 2, "unknown release safe");
    ++cases;
    SelfDriveSizes good;
    good.in_w = good.in_h = good.out_w = good.out_h = 64;
    good.valid = true;
    check(self_drive_armed(good, false) && !self_drive_armed(good, true),
          "self-drive needs template, no foreign driver");
    ++cases;
    SelfDriveSizes bad;
    check(!self_drive_armed(bad, false), "self-drive refuses without sizes");
    ++cases;
    check(kSelfDriveTryIds[0] == 0u && kSelfDriveTryIds[1] == 13u && kSelfDriveTryIds[2] == 18u,
          "self-drive tries SuperSampling NR vehicle, then RR ids");
    ++cases;
}
static void sl_shim_tests() {
    using namespace sl_shim;
    SlObserver observer;
    check(observer.size() == 0 && observer.find(1001) == nullptr &&
              observer.drive_template() == nullptr,
          "empty sl observer");
    ++cases;
    SlTag none[1] = {};
    observer.note_tags(7, none, 0, 0);
    check(observer.size() == 0 && observer.frames_held() == 0, "empty tags ignored");
    ++cases;
    SlTag frame7[4] = {};
    frame7[0] = {0x1000, kTagScalingInputColor, 1920, 1080, 10};
    frame7[1] = {0x2000, kTagMotion, 1920, 1080, 0};
    frame7[2] = {0x3000, kTagDepth, 1920, 1080, 0};
    frame7[3] = {0, kTagHUDLessColor, 64, 64, 0};
    observer.note_tags(7, frame7, 4, 0);
    observer.note_evaluate(1001, 7, 1920, 1080);
    const SlInputs* tpl = observer.drive_template();
    check(tpl && tpl->color_w == 1920 && tpl->color_h == 1080 && tpl->color_format == 10,
          "scaling input color wins");
    ++cases;
    check(tpl && tpl->motion && tpl->depth && tpl->out_w == 1920 && tpl->viewports == 1,
          "flags, output, viewport");
    ++cases;
    check(tpl && tpl->evaluates == 1 && tpl->tags == 3, "counts (null resource skipped)");
    ++cases;
    SlTag hud[1] = {};
    hud[0] = {0x4000, kTagHUDLessColor, 640, 384, 2};
    observer.note_tags(8, hud, 1, 0);
    observer.note_evaluate(1001, 8, 640, 384);
    tpl = observer.drive_template();
    check(tpl && tpl->color_w == 640 && tpl->color_format == 2, "hud-less fallback");
    ++cases;
    observer.note_evaluate(0, 8, 64, 64);
    observer.note_evaluate(7, 9, 0, 0);
    check(observer.drive_template() && observer.drive_template()->color_w == 640,
          "non-RR never templates");
    ++cases;
    check(observer.find(7) && observer.find(7)->evaluates == 1 && observer.find(7)->color_w == 0,
          "geometry-less evaluates counted");
    ++cases;
    const auto sizes = negotiate_sl_sizes(observer.drive_template());
    check(sizes.valid && sizes.in_w == 640 && sizes.in_h == 384 && sizes.format == 2,
          "sl sizes mirror template");
    ++cases;
    check(!negotiate_sl_sizes(nullptr).valid, "no sl template refuses");
    ++cases;
    for (uint32_t frame_index = 10; frame_index < 24; ++frame_index) {
        observer.note_tags(frame_index, hud, 1, 0);
    }
    check(observer.frames_held() <= 8, "frame window pruned");
    ++cases;
    SlObserver generation_observer;
    SlTag generation_tags[2] = {};
    generation_tags[0] = {0x9000, kTagScalingInputColor, 1280, 720, 10};
    generation_tags[1] = {0x9100, kTagMotion, 1280, 720, 0};
    generation_observer.note_tags_global(generation_tags, 2);
    check(generation_observer.global_tags() == 2 && generation_observer.last_tag_frame() == 0,
          "global tags held outside frames");
    ++cases;
    generation_observer.note_evaluate(1001, 4242, 0, 0);
    const SlInputs* generation_inputs = generation_observer.drive_template();
    check(generation_inputs && generation_inputs->color_w == 1280 && generation_inputs->motion &&
              generation_inputs->last_frame == 4242,
          "global fallback templates");
    ++cases;
    check(generation_observer.features().size() == 1 &&
              generation_observer.features().begin()->second.evaluates == 1,
          "forensic iteration");
    ++cases;
    SlObserver invalid_observer;
    SlTag invalid_tags[2] = {};
    invalid_tags[0] = {0xa000, kTagScalingInputColor, 0, 0, 10, 1920, 1080};
    invalid_tags[1] = {0xa001, kTagMotion, 0, 0, 0, 1920, 1080};
    invalid_observer.note_tags(3, invalid_tags, 2, 0);
    invalid_observer.note_evaluate(1001, 3, 0, 0);
    const SlInputs* xtpl = invalid_observer.drive_template();
    check(xtpl && xtpl->color_w == 1920 && xtpl->color_h == 1080,
          "extent fallback geometries color");
    ++cases;
    const auto rows = invalid_observer.last_table();
    check(rows.size() == 2 && rows[0].type == kTagMotion && rows[0].ex_w == 1920,
          "raw table forensics");
    ++cases;
    SlObserver identity_observer;
    uint64_t color_resource = 0, motion_resource = 0, depth_resource = 0;
    check(!identity_observer.current_inputs(color_resource, motion_resource, depth_resource),
          "no frames, no inputs");
    ++cases;
    SlTag identity_tags[2] = {};
    identity_tags[0] = {0xb000, kTagHUDLessColor, 800, 600, 10};
    identity_observer.note_tags(5, identity_tags, 1, 0);
    check(!identity_observer.current_inputs(color_resource, motion_resource, depth_resource),
          "color without motion refused");
    ++cases;
    SlTag ct2[3] = {};
    ct2[0] = {0xb001, kTagScalingInputColor, 800, 600, 10};
    ct2[1] = {0xb002, kTagMotion, 800, 600, 0};
    ct2[2] = {0xb003, kTagDepth, 800, 600, 0};
    identity_observer.note_tags(6, ct2, 3, 0);
    check(identity_observer.current_inputs(color_resource, motion_resource, depth_resource) &&
              color_resource == 0xb001 && motion_resource == 0xb002 && depth_resource == 0xb003,
          "newest frame pointers");
    ++cases;
    check(selfdrive_try_id(0) == 0u && selfdrive_try_id(1) == 13u && selfdrive_try_id(2) == 18u &&
              selfdrive_try_id(3) == 0u,
          "try NR vehicle, RR ids, exhaust");
    ++cases;
    check(ngx_shim::selfdrive_retry_due(5000, 0) && !ngx_shim::selfdrive_retry_due(4999, 5000) &&
              ngx_shim::selfdrive_retry_due(9999, 5000),
          "retry cooldown gates transient self-drive failures");
    ++cases;
}
static void routing_defaults_tests() {
    // Section defaults mirror fresh-struct values: every overlay option
    // reaches its default through its section button.
    {
        phoenix::Settings settings;
        settings.uplift = false;
        settings.controls.enabled = true;
        settings.controls.tone = 2;
        settings.controls.structure = 2;
        settings.controls.advanced = true;
        settings.controls.auto_mask = true;
        settings.controls.skin = 2;
        settings.controls.motion_x = 2;
        settings.controls.motion_y = 2;
        settings.controls.depth = 0;
        settings.host_unjitter = true;
        phoenix::reset_neural_section(settings);
        check(settings.uplift && !settings.controls.enabled && settings.controls.tone == 1 &&
                  settings.controls.structure == 1 && !settings.controls.advanced &&
                  !settings.controls.auto_mask && settings.controls.skin == -1 &&
                  settings.controls.motion_x == 1 && settings.controls.motion_y == 1 &&
                  settings.controls.depth == -1 && !settings.host_unjitter,
              "neural section defaults");
        ++cases;
    }
    {
        phoenix::Settings settings;
        settings.native_options = true;
        settings.intensity = 2;
        settings.style = 2;
        settings.ui_correction = 1;
        settings.host_exposure = 2;
        settings.host_exposure_scale = 2;
        settings.host_globaltone = 2;
        settings.host_jitter = 0;
        settings.host_bridge = 0;
        settings.host_white_nits = 100;
        settings.host_hdr = 1;
        settings.nr_auto_exposure = false;
        phoenix::reset_native_section(settings);
        check(!settings.native_options && settings.intensity == 1 && settings.style == 0 &&
                  settings.ui_correction == 0 && settings.host_exposure == 1 &&
                  settings.host_exposure_scale == 1 && settings.host_globaltone == 1 &&
                  settings.host_jitter == 1 && settings.host_bridge == 1 &&
                  settings.host_white_nits == 500 && settings.host_hdr == 0 &&
                  settings.nr_auto_exposure,
              "native section defaults");
        ++cases;
    }
    {
        phoenix::Settings settings;
        settings.comparison = 4;
        settings.source_a = 0;
        settings.source_b = 0;
        settings.divider = .1f;
        settings.difference_gain = 64;
        settings.comparison_labels = false;
        settings.comparison_lines = false;
        phoenix::reset_comparison_section(settings);
        check(settings.comparison == 0 && settings.source_a == 1 && settings.source_b == 2 &&
                  settings.divider == 2.f / 3.f && settings.difference_gain == 1 &&
                  settings.comparison_labels && settings.comparison_lines,
              "comparison section defaults");
        ++cases;
    }
    {
        phoenix::Settings settings;
        settings.force_reset = true;
        settings.reset_after_switch = 120;
        settings.reset_every = 600;
        phoenix::reset_history_section(settings);
        check(!settings.force_reset && settings.reset_after_switch == 1 &&
                  settings.reset_every == 0,
              "history section defaults");
        ++cases;
    }
    // RenoDX reference: the DLL comparison baseline, independent of the
    // struct defaults (explicit correction on, masks on, strengths at 1.0).
    {
        phoenix::Settings settings;
        settings.intensity = 2;
        settings.controls.tone = 2;
        settings.controls.structure = 2;
        settings.controls.skin = 2;
        settings.controls.motion_x = 2;
        settings.controls.depth = 0;
        settings.host_exposure = 2;
        settings.host_globaltone = 2;
        settings.host_jitter = 0;
        settings.uplift = false;
        phoenix::apply_nr_reference(settings);
        check(settings.uplift && settings.intensity == 1 && settings.controls.tone == 1 &&
                  settings.controls.structure == 1 && settings.controls.skin == -1 &&
                  settings.controls.motion_x == 1 && settings.controls.motion_y == 1 &&
                  settings.controls.depth == -1 && settings.controls.auto_mask &&
                  settings.native_options && settings.style == 0 && settings.ui_correction == 1 &&
                  settings.host_exposure == 1 && settings.host_exposure_scale == 1 &&
                  settings.host_globaltone == 1 && settings.host_jitter == 1 && settings.valid(),
              "RenoDX reference values validate");
        ++cases;
    }
    // New contract flags round-trip through overrides and validate.
    {
        phoenix::Settings settings;
        unsigned applied = 0, ignored = 0;
        check(settings.host_hdr == 0 && settings.nr_auto_exposure && settings.capture_burst == 0 &&
                  !settings.host_unjitter,
              "contract flag defaults");
        ++cases;
        check(phoenix::merge_overrides(
                  "[Phoenix]\nHostHdr=1\nNRAutoExposure=false\nHostUnjitter=false\n", settings,
                  applied, ignored) &&
                  applied == 3 && settings.host_hdr == 1 && !settings.nr_auto_exposure &&
                  !settings.host_unjitter && settings.valid(),
              "contract flags parse");
        ++cases;
        check(!phoenix::merge_overrides("[Phoenix]\nHostHdr=2\n", settings, applied, ignored) &&
                  settings.host_hdr == 1,
              "host hdr bound enforced");
        ++cases;
        check(phoenix::merge_overrides("[Phoenix]\nCaptureBurst=8\n", settings, applied, ignored) &&
                  applied == 1 && settings.capture_burst == 8 && settings.valid(),
              "capture burst parses");
        ++cases;
        check(
            phoenix::merge_overrides("[Phoenix]\nCaptureBurst=64\n", settings, applied, ignored) &&
                applied == 1 && settings.capture_burst == 64 && settings.valid(),
            "long capture burst parses");
        ++cases;
        check(
            !phoenix::merge_overrides("[Phoenix]\nCaptureBurst=65\n", settings, applied, ignored) &&
                settings.capture_burst == 64,
            "capture burst bound enforced");
        ++cases;
        check(
            phoenix::merge_overrides(
                "[Phoenix]\nBatchedDispatch=0\nRelaxBarriers=true\nMinimalBarriers=1\nNrResolveSteps=0\n",
                settings, applied, ignored) &&
                applied == 0 && ignored == 4 && settings.valid(),
            "production ignores obsolete scheduling and partial-render options");
        ++cases;
        std::string serialized;
        phoenix::visit_settings(settings, [&](const char* name, const auto&) {
            serialized += std::string(name) + "\n";
        });
        check(serialized.find("BatchedDispatch") == std::string::npos &&
                  serialized.find("RelaxBarriers") == std::string::npos &&
                  serialized.find("MinimalBarriers") == std::string::npos &&
                  serialized.find("NrResolveSteps") == std::string::npos,
              "production settings do not expose diagnostic controls");
        ++cases;
        check(settings.nr_feature == 18, "nr feature defaults to dlssnr 18");
        ++cases;
        check(phoenix::merge_overrides("[Phoenix]\nNrFeature=13\n", settings, applied, ignored) &&
                  applied == 1 && settings.nr_feature == 13 && settings.valid(),
              "nr feature 13 parses");
        ++cases;
        check(!phoenix::merge_overrides("[Phoenix]\nNrFeature=7\n", settings, applied, ignored) &&
                  settings.nr_feature == 13,
              "nr feature rejects garbage");
        ++cases;
        check(phoenix::merge_overrides("[Phoenix]\nScreenshotRequest=103\n", settings, applied,
                                       ignored) &&
                  applied == 1 && ignored == 0 && settings.screenshot_request == 103,
              "screenshot request uses shared settings parser");
        ++cases;
    }
}

static void ledger_tests() {
    using namespace phoenix_addon::ledger;
    constexpr uint64_t list_a = 0x1000, list_b = 0x2000, queue = 0x9000, other_queue = 0x9100;
    constexpr uint64_t colour = 0x10, output = 0x20;
    Ledger ledger;
    check(!ledger.state(output, list_a, queue).known, "unknown resource has no state");
    ++cases;
    ledger.created(output, 8);
    auto answer = ledger.state(output, list_a, queue);
    check(answer.known && answer.state == 8 && answer.basis == Basis::creation,
          "an untransitioned resource keeps its creation state");
    ++cases;
    ledger.transition(list_b, output, 0x400);
    check(!ledger.state(output, list_a, queue).known,
          "a transition held by another unsubmitted recording blocks any answer");
    ++cases;
    ledger.transition(list_a, output, 0x40);
    answer = ledger.state(output, list_a, queue);
    check(answer.known && answer.state == 0x40 && answer.basis == Basis::current_recording,
          "this recording's own transition wins");
    ++cases;
    ledger.reset(list_b);
    ledger.reset(list_a);
    check(!ledger.state(output, list_a, queue).known,
          "an abandoned recording establishes nothing and the creation state is spent");
    ++cases;
    ledger.transition(list_b, output, 0x800);
    ledger.submitted(list_b, queue);
    answer = ledger.state(output, list_a, queue);
    check(answer.known && answer.state == 0x800 && answer.basis == Basis::submitted_transition,
          "a submitted transition on the same queue is ordered before this recording");
    ++cases;
    check(!ledger.state(output, list_a, other_queue).known,
          "a transition submitted on another queue is not ordered against this one");
    ++cases;
    ledger.created(colour, 8);
    ledger.transition(list_b, colour, 0x40);
    ledger.submitted(list_b, queue);
    ledger.created(output, 8);
    answer = ledger.state(output, list_a, queue);
    check(answer.known && answer.state == 8 && answer.basis == Basis::creation,
          "a recycled handle must not inherit the previous resource's history");
    ++cases;
    ledger.destroyed(colour);
    check(!ledger.state(colour, list_a, queue).known, "a destroyed resource is forgotten");
    ++cases;
    ledger.transition(list_a, output, 0x400);
    ledger.transition(list_b, output, 0x40);
    ledger.submitted(list_b, queue);
    check(!ledger.state(output, list_b, queue).known,
          "a submitted transition is still blocked while another recording holds one");
    ++cases;
    ledger.destroyed_list(list_a);
    answer = ledger.state(output, list_b, queue);
    check(answer.known && answer.state == 0x40 && answer.basis == Basis::submitted_transition,
          "a destroyed recording releases its pending claim");
    ++cases;
    rejected([&] { ledger.transition(0, output, 8); });
    rejected([&] { ledger.created(0, 8); });
    ledger.created(colour, 8);
    ledger.transition(list_a, colour, 0x40);
    ledger.transition(list_b, colour, 0x400);
    ledger.destroyed(colour);
    check(!ledger.state(colour, list_a, queue).known && ledger.destroyed_pending_entries() == 2,
          "destroy purges all open references rather than leaking tombstones");
    ++cases;
    ledger.submitted(list_a, queue);
    ledger.submitted(list_b, queue);
    check(!ledger.state(colour, list_a, queue).known,
          "late submit cannot resurrect a destroyed resource");
    ++cases;
    ledger.created(colour, 8);
    ledger.transition(list_a, colour, 0x40);
    ledger.created(colour, 0x800);
    ledger.submitted(list_a, queue);
    check(ledger.state(colour, list_b, queue).state == 0x800 &&
              ledger.state(colour, list_b, queue).basis == Basis::creation,
          "recycled handle cannot inherit an old unsubmitted recording");
    ++cases;
    ledger.transition(list_a, colour, 0x40);
    ledger.submitted(list_a, queue);
    ledger.destroyed_queue(queue);
    check(!ledger.state(colour, list_b, queue).known,
          "reused queue pointer cannot inherit old queue ordering");
    ++cases;
    ledger.transition(list_a, colour, 0x400);
    ledger.submitted(list_a, queue);
    check(ledger.state(colour, list_b, queue).known &&
              ledger.state(colour, list_b, queue).state == 0x400,
          "new submission may re-establish ordering after queue replacement");
    ++cases;
    // explain() names the unknown-cause without mutating anything: untracked,
    // open-here, blocked-by-others, committed-elsewhere each read distinctly.
    {
        Ledger explainable;
        auto none = explainable.explain(0x50, list_a, queue);
        check(!none.tracked && none.transitions == 0 && none.open_recording_lists == 0,
              "explain untracked");
        ++cases;
        explainable.transition(list_b, 0x50, 0x40);
        auto elsewhere = explainable.explain(0x50, list_a, queue);
        check(elsewhere.tracked && !elsewhere.open_in_this_list &&
                  elsewhere.other_open_recordings == 1 && !elsewhere.committed &&
                  elsewhere.transitions == 1,
              "explain blocked-by-other");
        ++cases;
        explainable.transition(list_a, 0x50, 0x400);
        auto here = explainable.explain(0x50, list_a, queue);
        check(here.open_in_this_list && here.other_open_recordings == 1,
              "explain open-here keeps others");
        ++cases;
        explainable.submitted(list_b, queue);
        auto committed = explainable.explain(0x50, list_a, other_queue);
        check(committed.committed && !committed.committed_queue_matches,
              "explain committed-elsewhere");
        ++cases;
        auto mine = explainable.explain(0x50, list_a, queue);
        check(mine.open_in_this_list && mine.other_open_recordings == 0,
              "explain submitted-other releases block");
        ++cases;
        check(explainable.state(0x50, list_a, queue).known,
              "explain scenario still answers current-recording");
        ++cases;
    }
}
static void manifest_tests() {
    using trace::ManifestRole;
    // The b14 shape: 184 frames spooled, 185 submitted (quit before the last
    // drain), all three roles known. Byte-exact golden: the replay tool
    // parses this, so the layout is a contract, not cosmetics.
    std::vector<ManifestRole> roles(3);
    roles[0].name = "color";
    roles[0].known = true;
    roles[0].format = 26;
    roles[0].width = 2560;
    roles[0].height = 1080;
    roles[0].row_pitch = 10240;
    roles[0].num_rows = 1080;
    roles[0].total_bytes = 11059200;
    roles[1].name = "motion";
    roles[1].known = true;
    roles[1].format = 34;
    roles[1].width = 2560;
    roles[1].height = 1080;
    roles[1].row_pitch = 10240;
    roles[1].num_rows = 1080;
    roles[1].total_bytes = 11059200;
    roles[2].name = "output";
    roles[2].known = true;
    roles[2].format = 26;
    roles[2].width = 2560;
    roles[2].height = 1080;
    roles[2].row_pitch = 10240;
    roles[2].num_rows = 1080;
    roles[2].total_bytes = 11059200;
    const std::string golden =
        "{\"schema\":\"production-game-phoenix-trace/1\",\"utc\":\"20261002T085156Z\","
        "\"frames\":184,\"submitted\":185,\"dropped\":0,\"gaps\":0,\"roles\":["
        "{\"name\":\"color\",\"known\":true,\"format\":26,\"width\":2560,\"height\":1080,"
        "\"row_pitch\":10240,\"num_rows\":1080,\"total_bytes\":11059200},"
        "{\"name\":\"motion\",\"known\":true,\"format\":34,\"width\":2560,\"height\":1080,"
        "\"row_pitch\":10240,\"num_rows\":1080,\"total_bytes\":11059200},"
        "{\"name\":\"output\",\"known\":true,\"format\":26,\"width\":2560,\"height\":1080,"
        "\"row_pitch\":10240,\"num_rows\":1080,\"total_bytes\":11059200}],\"note\":\"operator\"}\n";
    check(trace::build_manifest_json("20261002T085156Z", 184, 185, 0, 0, roles, "operator") ==
              golden,
          "trace manifest golden incl completed-vs-submitted split");
    ++cases;
    // Unknown roles omit geometry; empty trace still serializes.
    std::vector<ManifestRole> partial(2);
    partial[0].name = "color";
    partial[0].known = true;
    partial[0].format = 26;
    partial[0].width = 64;
    partial[0].height = 64;
    partial[0].row_pitch = 256;
    partial[0].num_rows = 64;
    partial[0].total_bytes = 16384;
    partial[1].name = "output";
    const std::string small = trace::build_manifest_json("t", 0, 0, 1, 2, partial, "");
    check(small.find("\"frames\":0,\"submitted\":0,\"dropped\":1,\"gaps\":2") !=
                  std::string::npos &&
              small.find("{\"name\":\"output\",\"known\":false}") != std::string::npos &&
              small.find("\"note\":\"\"") != std::string::npos,
          "unknown roles omit geometry; counters and empty note serialize");
    ++cases;
}

static void dispatch_log_tests() {
    using dispatch_log::parse_log_frames;
    struct ParseCase {
        const char* text;
        unsigned expect;
    };
    const ParseCase parse[]{
        {nullptr, 0},
        {"", 0},
        {"300", 300},
        {"0", 0},
        {"abc", 0},
        {"-5", 0},
        {" -5", 0},
        {"+300", 300},
        {"100000", 100000},
        {"100001", 100000},
        {"99999999999999999999", 100000},
    };
    for (const auto& case_entry : parse) {
        check(parse_log_frames(case_entry.text) == case_entry.expect, "dispatch frame cap parse");
        ++cases;
    }
    check(std::string(dispatch_log::mh_status_name(0)) == "ok", "mh ok");
    ++cases;
    check(std::string(dispatch_log::mh_status_name(3)) == "already-created", "mh already-created");
    ++cases;
    check(std::string(dispatch_log::mh_status_name(8)) == "unsupported-function", "mh unsupported");
    ++cases;
    check(std::string(dispatch_log::mh_status_name(-1)) == "unknown", "mh unknown");
    ++cases;
    check(std::string(dispatch_log::mh_status_name(99)) == "unknown", "mh out-of-range");
    ++cases;
    check(dispatch_log::hex_bytes(nullptr, 0).empty(), "hex of empty is empty");
    ++cases;
    const unsigned char sample[3]{0x00, 0xff, 0x10};
    check(dispatch_log::hex_bytes(sample, 3) == "00ff10", "hex encodes nibbles");
    ++cases;
    check(dispatch_log::hex_bytes(nullptr, 5).empty(), "hex of null is empty");
    ++cases;
    check(dispatch_log::json_escape("a\"b\\c") == "a\\\"b\\\\c", "escape quotes/backslash");
    ++cases;
    check(dispatch_log::json_escape(std::string("x\x01y")) == "x\\u0001y", "escape control char");
    ++cases;
    const unsigned grid[3]{80, 34, 1}, block[3]{32, 8, 1};
    const unsigned char args[4]{0x9a, 0x3f, 0x00, 0xff};
    const std::string golden =
        "{\"f\":12,\"side\":\"dll\",\"seq\":3,\"name\":\"cc_split_swin_16h_qkv_512_chained_fp8\","
        "\"fn\":\"0x1a2b3c\",\"grid\":[80,34,1],\"block\":[32,8,1],\"shared\":0,\"params\":4,"
        "\"args\":\"9a3f00ff\"}\n";
    check(dispatch_log::format_launch_line(12, 0, 3, "cc_split_swin_16h_qkv_512_chained_fp8",
                                           0x1a2b3c, grid, block, 0, 4, args) == golden,
          "dispatch line golden");
    ++cases;
    std::vector<unsigned char> big(5000, 0xab);
    const std::string cut =
        dispatch_log::format_launch_line(7, 1, 0, nullptr, 0, grid, block, 1024, 5000, big.data());
    check(cut.find("\"side\":\"ours\"") != std::string::npos &&
              cut.find("\"name\":\"unknown\"") != std::string::npos &&
              cut.find("\"trunc\":true") != std::string::npos &&
              cut.find("\"args\":\"") != std::string::npos && cut.size() > 8192 &&
              cut.size() < 9000,
          "ours side, null name, oversize args truncate");
    ++cases;
    check(dispatch_log::format_launch_line(0, 2, 0, "n", 0, grid, block, 0, 0, nullptr)
                  .find("\"side\":\"ours\"") != std::string::npos,
          "nonzero side formats as ours");
    ++cases;
}

static void batch_tests() {
    struct ParseCase {
        const char* text;
        bool expect;
    };
    const ParseCase parse[]{
        {nullptr, false}, {"", false},       {"group", true},    {"GROUP", false},
        {"Group", false}, {" group", false}, {"group ", false},  {"1", false},
        {"on", false},    {"all", false},    {"batched", false},
    };
    for (const auto& case_entry : parse) {
        check(parse_batch_group(case_entry.text) == case_entry.expect, "batch group parse");
        ++cases;
    }
}

static void cubins_tests() {
    struct ParseCase {
        const char* text;
        bool expect;
    };
    const ParseCase parse[]{
        {nullptr, false},  {"", false},          {"native", true},
        {"NATIVE", false}, {"candidate", false}, {"cand", false},
    };
    for (const auto& case_entry : parse) {
        check(parse_native_cubins(case_entry.text) == case_entry.expect, "cubins parse");
        ++cases;
    }
}

static void skip_tests() {
    struct ParseCase {
        const char* text;
        uint32_t expect;
    };
    const ParseCase parse[]{
        {nullptr, 0},
        {"", 0},
        {"stem", 1u << 0},
        {"vit_1024", 1u << 7},
        {"head", 1u << 14},
        {"0", 1u << 0},
        {"7", 1u << 7},
        {"14", 1u << 14},
        {"encoder_128,decoder_128", (1u << 3) | (1u << 11)},
        {"3,11", (1u << 3) | (1u << 11)},
        {"vit_1024,7", 1u << 7},
        {"stem,encoder_32,encoder_64,encoder_128,encoder_256,encoder_split_512,repack_2d_to_1d,vit_1024,"
         "repack_1d_to_2d,decoder_split_512,decoder_256,decoder_128,decoder_64,decoder_32,head",
         0x7fffu},
    };
    for (const auto& case_entry : parse) {
        check(parse_skip_groups(case_entry.text) == case_entry.expect, "skip groups parse");
        ++cases;
    }
    const char* reject[]{
        " ",   "vit_1024 ", " vit_1024", "vit_1024, decoder_128", "vit",     "VIT_1024", "15",
        "-1",  "7,",        ",7",        "vit_1024,,decoder_128", "encoder", "16",       "0x7",
        "7.0",
    };
    for (const auto& text : reject) {
        rejected([&] { parse_skip_groups(text); });
    }
    check(kGroupCount == 15 && kGroupEnds[kGroupCount - 1] == kLaunchCount && kLaunchCount == 154,
          "skip schedule constants");
    ++cases;
    for (unsigned iteration_index = 1; iteration_index < kGroupCount; ++iteration_index) {
        check(kGroupEnds[iteration_index] > kGroupEnds[iteration_index - 1], "skip ends increase");
        ++cases;
    }
    check(group_of_position(0) == 0 && group_of_position(1) == 1 && group_of_position(55) == 5 &&
              group_of_position(56) == 6 && group_of_position(57) == 7 &&
              group_of_position(153) == 14,
          "skip position groups");
    ++cases;
    check(skipped_launches(0) == 0, "skip none");
    ++cases;
    check(skipped_launches(1u << 7) == 40, "skip vit launches");
    ++cases;
    check(skipped_launches((1u << 3) | (1u << 11)) == 12, "skip 128 launches");
    ++cases;
    check(skipped_launches(0x7fffu) == 154, "skip all launches");
    ++cases;
    ChainAccount acc;
    acc.add(154);
    check(acc.kernels == 154 && acc.chains == 1, "chain account sums one evaluate");
    ++cases;
    acc.add(0);
    acc.add(1000000u);
    check(acc.kernels == 154 && acc.chains == 1, "chain account ignores insane counts");
    ++cases;
    acc.kernels = UINT64_MAX - 10;
    acc.add(154);
    check(acc.kernels == UINT64_MAX, "chain account saturates instead of wrapping");
    ++cases;
    acc.reset();
    check(acc.kernels == 0 && acc.chains == 0, "chain account resets per call");
    ++cases;
    check(budget_ema_update(0, 8000) == 1000 && budget_ema_update(8000, 8000) == 8000,
          "budget ema converges");
    ++cases;
    check(budget_should_skip(9000, 8000) && !budget_should_skip(7000, 8000),
          "budget skips over cap");
    ++cases;
    check(!budget_should_skip(UINT64_MAX, 0), "budget zero never skips");
    ++cases;
    check(ablation_skips_chain(0x1u, 0) && !ablation_skips_chain(0x1u, 1) &&
              !ablation_skips_chain(0u, 0),
          "ablation bit selects chain position");
    ++cases;
    check(ablation_skips_chain(0xFFFFFFFFu, 31) && !ablation_skips_chain(0xFFFFFFFFu, 32),
          "ablation bounds at 32");
    ++cases;
    const unsigned char elf[64] = {0x7f, 'E', 'L', 'F'};
    const unsigned char txt[64] = {'M', 'Z'};
    const unsigned char tiny[4] = {0x7f, 'E', 'L', 'F'};
    check(looks_like_cubin(elf, sizeof(elf)) && !looks_like_cubin(txt, sizeof(txt)),
          "cubin magic accepted, non-elf refused");
    ++cases;
    check(!looks_like_cubin(tiny, sizeof(tiny)) && !looks_like_cubin(nullptr, 128),
          "cubin rejects tiny/null");
    ++cases;
}

static void panel_tests() {
    using namespace panel;
    check(kBackendCount == 4, "panel backend count");
    ++cases;
    check(std::string(kBackendLabels[0]) == "Game native (NR off)" &&
              std::string(kBackendLabels[3]) == "CNR ours (fullscreen)",
          "panel backend labels");
    ++cases;
    check(kSkipPresetCount == 3 && kSkipPresetMasks[0] == 0 &&
              kSkipPresetMasks[1] == ((1u << 3) | (1u << 11)) && kSkipPresetMasks[2] == (1u << 7),
          "panel skip preset masks");
    ++cases;
    check(skipped_launches(kSkipPresetMasks[1]) == 12, "panel 128 preset launches");
    ++cases;
    check(skipped_launches(kSkipPresetMasks[2]) == 40, "panel vit preset launches");
    ++cases;
    check(kUicValues[0] == -1 && kUicValues[1] == 0 && kUicValues[2] == 1, "panel uic values");
    ++cases;
    check(cycle_skip(0) == 1 && cycle_skip(1) == 2 && cycle_skip(2) == 0, "panel skip cycle");
    ++cases;
    check(cycle_backend(3) == 0 && cycle_backend(1) == 2, "panel backend cycle");
    ++cases;
    check(cycle_uic(-1) == 0 && cycle_uic(0) == 1 && cycle_uic(1) == -1, "panel uic cycle");
    ++cases;
    MsHistory empty;
    check(empty.size() == 0 && empty.average() == 0.0 && empty.ordered_min() == 0.0f &&
              empty.ordered_max() == 0.0f,
          "panel empty history");
    ++cases;
    MsHistory history;
    history.push(1.0);
    history.push(3.0);
    check(history.size() == 2 && history.average() == 2.0 && history.ordered_min() == 1.0f &&
              history.ordered_max() == 3.0f,
          "panel history stats");
    ++cases;
    history.push(-1.0);
    history.push(1000.5);
    check(history.size() == 2, "panel history drops bogus pairs");
    ++cases;
    MsHistory wrapped;
    for (unsigned iteration_index = 1; iteration_index <= MsHistory::kCapacity + 10;
         ++iteration_index)
        wrapped.push(double(iteration_index));
    float ordered[MsHistory::kCapacity]{};
    wrapped.copy_ordered(ordered);
    check(wrapped.size() == MsHistory::kCapacity && ordered[0] == 11.0f &&
              ordered[MsHistory::kCapacity - 1] == float(MsHistory::kCapacity + 10),
          "panel history wraps oldest-first");
    ++cases;
    check(wrapped.ordered_min() == 11.0f &&
              wrapped.ordered_max() == float(MsHistory::kCapacity + 10),
          "panel wrapped min/max");
    ++cases;
    const auto defaults = parse_ini(nullptr);
    check(defaults.backend == 1 && defaults.skip == 0 && defaults.uic == -1 && !defaults.visible,
          "panel ini null defaults");
    ++cases;
    const auto from_empty = parse_ini("");
    check(from_empty.backend == 1 && !from_empty.visible, "panel ini empty defaults");
    ++cases;
    const auto full = parse_ini("backend=2\nskip=1\nuic=0\nvisible=1\n");
    check(full.backend == 2 && full.skip == 1 && full.uic == 0 && full.visible, "panel ini full");
    ++cases;
    const auto unknown = parse_ini("backend=3\nupscaler=fsr\nvisible=1\n");
    check(unknown.backend == 3 && unknown.visible && unknown.skip == 0,
          "panel ini ignores unknown keys");
    ++cases;
    const auto malformed = parse_ini("backend\nskip=7\nskip=x\nuic=2\nvisible=9\nbackend=0\n");
    check(malformed.backend == 0 && malformed.skip == 0 && malformed.uic == -1 &&
              !malformed.visible,
          "panel ini ignores malformed values");
    ++cases;
    const auto spaced = parse_ini("# comment\n; other\n[section]\n  skip = 2 \r\nvisible=1");
    check(spaced.skip == 2 && spaced.visible && spaced.backend == 1,
          "panel ini comments and spacing");
    ++cases;
    PanelConfig custom;
    custom.backend = 0;
    custom.skip = 2;
    custom.uic = 1;
    custom.visible = true;
    const auto round = parse_ini(serialize_ini(custom).c_str());
    check(round.backend == 0 && round.skip == 2 && round.uic == 1 && round.visible,
          "panel ini round trip");
    ++cases;
    check(serialize_ini(PanelConfig{}) ==
              "backend=1\nskip=0\nuic=-1\nvisible=0\n"
              "nr_uplift=1\nnr_style=0\nnr_automask=1\nnr_uicorr=1\nnr_depth_mode=0\n"
              "nr_intensity=1.000000\nnr_structure=0.990000\nnr_tone=0.450000\nnr_skin=0.990000\n"
              "nr_paper_white=1.000000\nnr_transfer_strength=1.000000\nnr_color_strength=1.000000\n"
              "nr_mvec_scale_x=1.000000\nnr_mvec_scale_y=1.000000\n",
          "panel ini canonical defaults");
    ++cases;
    const auto nr_defaults = PanelConfig{};
    check(nr_defaults.uplift == 1 && nr_defaults.style == 0 && nr_defaults.automask == 1 &&
              nr_defaults.uicorr == 1 && nr_defaults.depth_mode == 0 &&
              nr_defaults.intensity == 1.0f && nr_defaults.structure == 0.99f &&
              nr_defaults.tone == 0.45f && nr_defaults.skin == 0.99f &&
              nr_defaults.paper_white == 1.0f && nr_defaults.transfer_strength == 1.0f &&
              nr_defaults.color_strength == 1.0f && nr_defaults.mvec_scale_x == 1.0f &&
              nr_defaults.mvec_scale_y == 1.0f,
          "panel nr defaults match the feeder");
    ++cases;
    const auto nr_full = parse_ini("nr_uplift=0\nnr_style=1\nnr_automask=0\nnr_uicorr=0\n"
                                   "nr_depth_mode=2\nnr_intensity=1\nnr_structure=0.5\n"
                                   "nr_tone=1.25\nnr_skin=2\nnr_paper_white=16\n"
                                   "nr_transfer_strength=0\nnr_color_strength=1.5\n"
                                   "nr_mvec_scale_x=4\nnr_mvec_scale_y=0.25\n");
    check(nr_full.uplift == 0 && nr_full.style == 1 && nr_full.automask == 0 &&
              nr_full.uicorr == 0 && nr_full.depth_mode == 2 && nr_full.intensity == 1.0f &&
              nr_full.structure == 0.5f && nr_full.tone == 1.25f && nr_full.skin == 2.0f &&
              nr_full.paper_white == 16.0f && nr_full.transfer_strength == 0.0f &&
              nr_full.color_strength == 1.5f && nr_full.mvec_scale_x == 4.0f &&
              nr_full.mvec_scale_y == 0.25f,
          "panel nr full parse, integral float text included");
    ++cases;
    const auto nr_bad = parse_ini("nr_uplift=2\nnr_style=-1\nnr_automask=7\nnr_uicorr=x\n"
                                  "nr_depth_mode=3\nnr_intensity=2.5\nnr_structure=-1\n"
                                  "nr_tone=abc\nnr_skin=inf\nnr_paper_white=17\n"
                                  "nr_transfer_strength=nan\nnr_color_strength=1.0.0\n"
                                  "nr_mvec_scale_x=\nnr_mvec_scale_y=5\n");
    check(nr_bad.uplift == 1 && nr_bad.style == 0 && nr_bad.automask == 1 && nr_bad.uicorr == 1 &&
              nr_bad.depth_mode == 0 && nr_bad.intensity == 1.0f && nr_bad.structure == 0.99f &&
              nr_bad.tone == 0.45f && nr_bad.skin == 0.99f && nr_bad.paper_white == 1.0f &&
              nr_bad.transfer_strength == 1.0f && nr_bad.color_strength == 1.0f &&
              nr_bad.mvec_scale_x == 1.0f && nr_bad.mvec_scale_y == 1.0f,
          "panel nr ignores malformed and out-of-range values");
    ++cases;
    const auto nr_style2 = parse_ini("nr_style=2\nnr_intensity=1\n");
    check(nr_style2.style == 2 && nr_style2.intensity == 1.0f, "panel nr cinematic style accepted");
    ++cases;
    const auto nr_style3 = parse_ini("nr_style=3\n");
    check(nr_style3.style == 0, "panel nr style bound enforced");
    ++cases;
    PanelConfig nr_custom;
    nr_custom.uplift = 0;
    nr_custom.style = 1;
    nr_custom.automask = 0;
    nr_custom.uicorr = 0;
    nr_custom.depth_mode = 1;
    nr_custom.intensity = 1.5f;
    nr_custom.structure = 0.1f;
    nr_custom.tone = 1.9f;
    nr_custom.skin = 0.2f;
    nr_custom.paper_white = 12.5f;
    nr_custom.transfer_strength = 0.3f;
    nr_custom.color_strength = 1.7f;
    nr_custom.mvec_scale_x = 3.5f;
    nr_custom.mvec_scale_y = 0.5f;
    const auto nr_round = parse_ini(serialize_ini(nr_custom).c_str());
    const auto near = [](float left, float right) {
        const float difference = left > right ? left - right : right - left;
        return difference < 0.00001f;
    };
    check(nr_round.uplift == 0 && nr_round.style == 1 && nr_round.automask == 0 &&
              nr_round.uicorr == 0 && nr_round.depth_mode == 1 && near(nr_round.intensity, 1.5f) &&
              near(nr_round.structure, 0.1f) && near(nr_round.tone, 1.9f) &&
              near(nr_round.skin, 0.2f) && near(nr_round.paper_white, 12.5f) &&
              near(nr_round.transfer_strength, 0.3f) && near(nr_round.color_strength, 1.7f) &&
              near(nr_round.mvec_scale_x, 3.5f) && near(nr_round.mvec_scale_y, 0.5f),
          "panel nr round trip");
    ++cases;
    check(render_target_format(kDxgiR8G8B8A8UnormSrgb) == kDxgiR8G8B8A8Unorm,
          "panel srgb target renders through the linear view");
    ++cases;
    check(render_target_format(kDxgiR8G8B8A8Unorm) == kDxgiR8G8B8A8Unorm &&
              render_target_format(0) == 0 && render_target_format(10) == 10,
          "panel other targets render as-is");
    ++cases;
    OutputIdentity left, right;
    check(!output_changed(left, right), "panel identical output keeps its RTV");
    ++cases;
    right.width = 1920;
    check(output_changed(left, right), "panel resized output rebuilds its RTV");
    ++cases;
    right = left;
    right.resource = 1;
    right.format = kDxgiR8G8B8A8UnormSrgb;
    check(output_changed(left, right), "panel new resource or format rebuilds its RTV");
    ++cases;
    check(backend_from_name("game") == 0 && backend_from_name("nvidia") == 1 &&
              backend_from_name("ours") == 2 && backend_from_name("ours_full") == 3,
          "panel backend names map to indices");
    ++cases;
    check(backend_from_name(nullptr) == 1 && backend_from_name("") == 1 &&
              backend_from_name("fsr") == 1 && backend_from_name("OURS") == 1,
          "panel unknown backend names fall back to nvidia");
    ++cases;
}

static void observe_tests() {
    using phoenix::observe_config_valid;
    check(observe_config_valid(false, false, 0, false, 1), "observe minimal valid");
    ++cases;
    check(observe_config_valid(false, false, 0, false, 4096), "observe max valid");
    ++cases;
    check(observe_config_valid(false, true, 0, false, 128), "observe accepts zero routing period");
    ++cases;
    check(!observe_config_valid(true, false, 0, false, 128), "observe rejects claim");
    ++cases;
    check(!observe_config_valid(false, true, 8, false, 128), "observe rejects routing period");
    ++cases;
    check(!observe_config_valid(false, false, 0, true, 128), "observe rejects skip groups");
    ++cases;
    check(!observe_config_valid(false, false, 0, false, 0), "observe rejects zero cap");
    ++cases;
    check(!observe_config_valid(false, false, 0, false, 4097), "observe rejects over cap");
    ++cases;
}

int main() {
    try {
        struct NativeParameters {
            std::map<std::string, std::variant<int, unsigned, float>> values;
            unsigned sets = 0;
            bool fail_once = false;
            int Get(const char* key, int* out) const {
                auto lookup_iterator = values.find(key);
                if (lookup_iterator == values.end() ||
                    !std::holds_alternative<int>(lookup_iterator->second))
                    return -1;
                *out = std::get<int>(lookup_iterator->second);
                return 0;
            }
            int Get(const char* key, unsigned* out) const {
                auto lookup_iterator = values.find(key);
                if (lookup_iterator == values.end() ||
                    !std::holds_alternative<unsigned>(lookup_iterator->second))
                    return -1;
                *out = std::get<unsigned>(lookup_iterator->second);
                return 0;
            }
            int Get(const char* key, float* out) const {
                auto lookup_iterator = values.find(key);
                if (lookup_iterator == values.end() ||
                    !std::holds_alternative<float>(lookup_iterator->second))
                    return -1;
                *out = std::get<float>(lookup_iterator->second);
                return 0;
            }
            void Set(const char* key, int value) {
                ++sets;
                values[key] = value;
            }
            void Set(const char* key, unsigned value) {
                ++sets;
                values[key] = value;
            }
            void Set(const char* key, float value) {
                if (fail_once) {
                    fail_once = false;
                    throw std::runtime_error("injected setter failure");
                }
                ++sets;
                values[key] = value;
            }
        } native;
        native.values = {{"reset", 7}, {"tone", .75f}, {"style", 0u}};
        const auto original_values = native.values;
        auto success = [](int result) { return result == 0; };
        {
            phoenix::ParameterEdits edit(native, success);
            edit.add("reset", 1);
            edit.add("tone", .25f);
            edit.add("style", 1u);
            check(native.values == original_values && native.sets == 0, "preflight is read-only");
            ++cases;
            edit.apply();
            check(std::get<int>(native.values["reset"]) == 1 &&
                      std::get<float>(native.values["tone"]) == .25f,
                  "native receives overrides");
            ++cases;
            rejected([&] { edit.add("reset", 0); });
            edit.restore();
            check(native.values == original_values, "restore exact types and values");
            ++cases;
        }
        {
            phoenix::ParameterEdits edit(native, success);
            edit.add("reset", 1);
            edit.apply();
        }
        check(native.values == original_values, "scope exit restores native reset");
        ++cases;
        {
            phoenix::ParameterEdits edit(native, success);
            edit.add("reset", 1);
            rejected([&] { edit.add("missing", .5f); });
            check(native.values == original_values, "missing getter cannot partially override");
            ++cases;
        }
        {
            phoenix::ParameterEdits edit(native, success);
            edit.add("reset", 1);
            edit.add("tone", .5f);
            native.fail_once = true;
            rejected([&] { edit.apply(); });
            check(native.values == original_values, "setter failure rolls back preceding edits");
            ++cases;
        }
        for (int mode = 0; mode < 4; ++mode) {
            phoenix::Settings panel;
            panel.mode = mode;
            check(panel.valid(), "all four routes validate");
            ++cases;
            check(phoenix::uses_native(mode) == (mode == 1 || mode == 2) &&
                      phoenix::uses_ours(mode) == (mode == 2 || mode == 3),
                  "route execution matrix");
            ++cases;
        }
        {
            phoenix::Settings partial;
            partial.controls.tone = .7f;
            unsigned applied = 0, ignored = 0;
            check(
                phoenix::merge_overrides(
                    "[Other]\nBackend=0\n[Phoenix]\nComparison=4\nSourceA=2\nSourceB=0\nDifferenceGain=8\nForceReset=true\nLogLevel=2\nOops=3\nDivider=garbage\n",
                    partial, applied, ignored),
                "partial merge accepted");
            ++cases;
            check(applied == 6 && ignored == 2 && partial.mode == 3 && partial.comparison == 4 &&
                      partial.source_a == 2 && partial.source_b == 0 &&
                      partial.difference_gain == 8 && partial.force_reset &&
                      partial.log_level == 2 && partial.controls.tone == .7f,
                  "partial fields preserve omitted settings");
            ++cases;
            check(!phoenix::merge_overrides("[Phoenix]\nSourceA=3\n", partial, applied, ignored) &&
                      partial.source_a == 2,
                  "invalid transaction atomic");
            ++cases;
            check(phoenix::merge_overrides("[Phoenix]\nStyle=2\n", partial, applied, ignored) &&
                      applied == 1 && partial.style == 2,
                  "cinematic style accepted");
            ++cases;
            check(!phoenix::merge_overrides("[Phoenix]\nStyle=3\n", partial, applied, ignored) &&
                      partial.style == 2,
                  "style bound enforced");
            ++cases;
            check(phoenix::merge_overrides("[Phoenix]\nNeuralUplift=false\n", partial, applied,
                                           ignored) &&
                      applied == 1 && !partial.uplift,
                  "uplift master parses");
            ++cases;
            {
                phoenix::Settings fresh;
                check(fresh.uplift && fresh.style == 0 && fresh.intensity == 1.0f,
                      "uplift/style/intensity defaults");
                ++cases;
            }
            {
                phoenix::Settings fresh;
                check(fresh.host_exposure == 1.0f && fresh.host_exposure_scale == 1.0f &&
                          fresh.host_globaltone == 1.0f && fresh.host_jitter == 1,
                      "host recipe defaults");
                ++cases;
            }
            check(phoenix::merge_overrides(
                      "[Phoenix]\nHostExposure=2\nHostGlobalTone=1.5\nHostJitter=0\n", partial,
                      applied, ignored) &&
                      applied == 3 && partial.host_exposure == 2.0f &&
                      partial.host_globaltone == 1.5f && partial.host_jitter == 0,
                  "host recipe parses");
            ++cases;
            check(!phoenix::merge_overrides("[Phoenix]\nHostExposure=9\n", partial, applied,
                                            ignored) &&
                      partial.host_exposure == 2.0f,
                  "host exposure bound enforced");
            ++cases;
            check(
                !phoenix::merge_overrides("[Phoenix]\nHostJitter=2\n", partial, applied, ignored) &&
                    partial.host_jitter == 0,
                "host jitter bound enforced");
            ++cases;
            check(phoenix::host_strength(true, 2.0f, 1.0f) == 2.0f &&
                      phoenix::host_strength(false, 2.0f, 1.0f) == 1.0f &&
                      phoenix::host_strength(true, 2, 0) == 2 &&
                      phoenix::host_strength(false, 2, 0) == 0,
                  "host_strength falls back to reference");
            ++cases;
            check(!phoenix::wipe_uses_previous_frame(0) && phoenix::wipe_uses_previous_frame(1) &&
                      !phoenix::wipe_uses_previous_frame(2) &&
                      !phoenix::wipe_uses_previous_frame(5),
                  "wipe lags one frame, forensic modes stay current");
            ++cases;
            {
                phoenix::Settings fresh;
                check(fresh.host_bridge == 1 && fresh.host_white_nits == 500.0f,
                      "host bridge defaults");
                ++cases;
            }
            check(phoenix::merge_overrides("[Phoenix]\nHostBridge=0\nHostWhiteNits=100\n", partial,
                                           applied, ignored) &&
                      applied == 2 && partial.host_bridge == 0 && partial.host_white_nits == 100.0f,
                  "host bridge parses");
            ++cases;
            check(
                !phoenix::merge_overrides("[Phoenix]\nHostBridge=2\n", partial, applied, ignored) &&
                    partial.host_bridge == 0,
                "host bridge bound enforced");
            ++cases;
            check(!phoenix::merge_overrides("[Phoenix]\nHostWhiteNits=0\n", partial, applied,
                                            ignored) &&
                      partial.host_white_nits == 100.0f,
                  "host white bound enforced");
            ++cases;
            check(!phoenix::merge_overrides("[Phoenix]\nDifferenceGain=65\n", partial, applied,
                                            ignored),
                  "gain bounded");
            ++cases;
            check(phoenix::merge_overrides(
                      "[Phoenix]\nForceReset=garbage\nSourceA=1junk\nDifferenceGain=nan\n", partial,
                      applied, ignored) &&
                      applied == 0 && ignored == 3 && partial.force_reset,
                  "malformed numeric and bool ignored");
            ++cases;
            check(!phoenix::merge_overrides(std::string(65537, 'x'), partial, applied, ignored),
                  "override length bounded");
            ++cases;
            for (int mode = 0; mode < 6; ++mode)
                for (int source_index = 0; source_index < 3; ++source_index)
                    for (int destination_index = 0; destination_index < 3; ++destination_index) {
                        partial.comparison = mode;
                        partial.source_a = source_index;
                        partial.source_b = destination_index;
                        check(partial.valid(), "all comparison pairs accepted");
                        ++cases;
                    }
            partial.comparison = 6;
            check(!partial.valid(), "comparison mode bound");
            ++cases;
            partial = {};
            partial.difference_gain = std::numeric_limits<float>::quiet_NaN();
            check(!partial.valid(), "comparison finite gain");
            ++cases;
        }
        for (int mode = 0; mode < 4; ++mode)
            for (int view = 0; view < 6; ++view)
                for (unsigned flags = 0; flags < 64; ++flags) {
                    phoenix::Settings settings;
                    settings.mode = mode;
                    settings.comparison = view;
                    settings.controls.enabled = (flags & 1) != 0;
                    settings.controls.advanced = (flags & 2) != 0;
                    settings.native_options = (flags & 4) != 0;
                    settings.experimental = (flags & 8) != 0;
                    settings.automatic_budget = (flags & 16) != 0;
                    settings.force_reset = (flags & 32) != 0;
                    const phoenix::PanelVisibility visible{settings};
                    check(visible.neural() == (mode != 0) && visible.comparison() == (mode == 2) &&
                              visible.pair_sources() == (mode == 2 && view != 0) &&
                              visible.divider() == (mode == 2 && view < 2) &&
                              visible.difference_gain() == (mode == 2 && view >= 2) &&
                              visible.strengths() == (mode != 0 && bool(flags & 1)) &&
                              visible.guides() == (mode != 0 && bool(flags & 2)) &&
                              visible.native() == (mode == 1 || mode == 2) &&
                              visible.native_values() ==
                                  ((mode == 1 || mode == 2) && bool(flags & 4)) &&
                              visible.experiments() == (mode == 3) &&
                              visible.depth() == (mode == 3 && bool(flags & 8)) &&
                              visible.budget() ==
                                  (mode == 3 && bool(flags & 8) && bool(flags & 16)) &&
                              visible.history_cadence() == (mode != 0 && !(flags & 32)) &&
                              visible.own_details() == (mode == 2 || mode == 3) &&
                              visible.hotkey(0) && visible.hotkey(2) && visible.hotkey(4) &&
                              visible.hotkey(1) == (mode != 0) &&
                              visible.hotkey(3) == (mode == 3 && bool(flags & 8)),
                          "contextual panel visibility matrix");
                    ++cases;
                    settings.mode = 3;
                    settings.experimental = true;
                    settings.automatic_budget = true;
                    check(visible.budget(), "visibility reacts to same-frame changes");
                    ++cases;
                }
        phoenix::Settings bad;
        bad.mode = 4;
        check(!bad.valid(), "reject invalid backend");
        ++cases;
        for (int blocks = 1; blocks <= 8; ++blocks)
            for (int mode = 0; mode < 4; ++mode) {
                phoenix::Settings settings;
                settings.mode = mode;
                settings.transformer_blocks = blocks;
                check(settings.valid() && settings.blocks_for(mode) == 8,
                      "approximation default off");
                ++cases;
                settings.experimental = true;
                check(settings.blocks_for(mode) == unsigned(mode == 3 ? blocks : 8),
                      "depth restricted to DLSS5 Phoenix-only route");
                ++cases;
            }
        bad = {};
        bad.transformer_blocks = 0;
        check(!bad.valid(), "reject empty transformer prefix");
        ++cases;
        bad = {};
        bad.transformer_blocks = 9;
        check(!bad.valid(), "reject excessive transformer prefix");
        ++cases;
        bad = {};
        bad.transformer_blocks = 4;
        bad.minimum_blocks = 5;
        check(!bad.valid(), "budget floor exceeds ceiling");
        ++cases;
        bad = {};
        bad.budget_ms = std::numeric_limits<float>::quiet_NaN();
        check(!bad.valid(), "budget NaN rejected");
        ++cases;
        bad = {};
        bad.desired_fps = 0;
        check(!bad.valid(), "zero FPS target rejected");
        ++cases;
        bad = {};
        bad.desired_fps = 361;
        check(!bad.valid(), "excessive FPS target rejected");
        ++cases;
        bad = {};
        bad.desired_fps = std::numeric_limits<float>::quiet_NaN();
        check(!bad.valid(), "NaN FPS target rejected");
        ++cases;
        bad = {};
        bad.budget_mode = 2;
        check(!bad.valid(), "unknown budget mode rejected");
        ++cases;
        bad = {};
        bad.refinement_passes = 0;
        check(!bad.valid(), "zero refinement rejected");
        ++cases;
        bad = {};
        bad.refinement_passes = 5;
        check(!bad.valid(), "unbounded refinement rejected");
        ++cases;
        for (unsigned destination_index = 0; destination_index < 8; ++destination_index) {
            bad = {};
            bad.block_repeats[destination_index] = 0;
            check(bad.valid(), "block bypass accepted");
            ++cases;
            bad.block_repeats[destination_index] = -1;
            check(!bad.valid(), "negative repeated block rejected");
            ++cases;
            bad.block_repeats[destination_index] = 5;
            check(!bad.valid(), "unbounded repeated block rejected");
            ++cases;
            bad.block_repeats[destination_index] = 4;
            check(bad.valid(), "bounded repeat accepted");
            ++cases;
        }
        {
            phoenix::Settings settings;
            unsigned applied = 0, ignored = 0;
            check(phoenix::merge_overrides(
                      "[Phoenix]\nRefinementPasses=2\nBlock1Repeats=3\nBlock8Repeats=4\n", settings,
                      applied, ignored) &&
                      settings.refinement_passes == 2 && settings.block_repeats[0] == 3 &&
                      settings.block_repeats[7] == 4 && applied == 3,
                  "repeat profile persists");
            ++cases;
            check(phoenix::merge_overrides("[Phoenix]\nBlock5Repeats=0\n", settings, applied,
                                           ignored) &&
                      settings.block_repeats[4] == 0,
                  "zero repetition persists");
            ++cases;
            check(!phoenix::merge_overrides("[Phoenix]\nBlock5Repeats=-1\nRefinementPasses=3\n",
                                            settings, applied, ignored) &&
                      settings.refinement_passes == 2,
                  "invalid repetition profile is transactional");
            ++cases;
        }
        {
            phoenix::Settings settings;
            unsigned applied = 0, ignored = 0;
            check(phoenix::merge_overrides(
                      "[Phoenix]\nBudgetMode=1\nDesiredFPS=50\nAutomaticBudget=1\n", settings,
                      applied, ignored) &&
                      settings.budget_mode == 1 && settings.desired_fps == 50 &&
                      settings.budget_target_ms() == 20 && applied == 3,
                  "FPS mode persists through the common settings visitor");
            ++cases;
            check(!phoenix::merge_overrides("[Phoenix]\nDesiredFPS=0\n", settings, applied,
                                            ignored) &&
                      settings.desired_fps == 50,
                  "invalid FPS profile is transactional");
            ++cases;
            phoenix::DepthBudget budget;
            budget.configure(true, 8, 2, 1000.0 / 60.0, true);
            uint64_t tick = 1000;
            auto advance = [&](unsigned frames, unsigned interval, bool active = true) {
                for (unsigned iteration_index = 0; iteration_index < frames; ++iteration_index) {
                    tick += interval;
                    budget.present(budget.depth(), tick * 1000, active);
                }
            };
            for (unsigned iteration_index = 0; iteration_index < 90; ++iteration_index)
                budget.sample(8, 100);
            check(budget.depth() == 8, "FPS mode ignores neural GPU timing samples");
            ++cases;
            advance(41, 50);
            check(budget.depth() == 7,
                  "sustained low presentation FPS lowers one block after two windows");
            ++cases;
            for (unsigned iteration_index = 0; iteration_index < 100; ++iteration_index) {
                tick += 20;
                budget.present(8, tick * 1000, true);
            }
            check(budget.depth() == 7,
                  "old-depth presentation samples cannot trigger further changes");
            ++cases;
            advance(180, 100);
            check(budget.depth() == 2, "unattainable FPS target respects minimum depth");
            ++cases;
            advance(200, 16, false);
            check(budget.depth() == 2 && budget.ema() == 0,
                  "unfocused or inactive inference discards FPS samples");
            ++cases;
            // Exact 50 FPS cap still permits cautious quality recovery.
            budget.configure(true, 8, 2, 20, true);
            advance(41, 50);
            check(budget.depth() == 7, "new FPS target starts a fresh observation window");
            ++cases;
            advance(251, 20);
            check(budget.depth() == 8, "meeting a frame cap restores quality after cooldown");
            ++cases;
            advance(41, 50);
            const auto before = budget.depth();
            for (unsigned iteration_index = 0; iteration_index < 40; ++iteration_index) {
                tick += 1000;
                budget.present(budget.depth(), tick * 1000, true);
            }
            check(budget.depth() == before, "pauses do not consume quality");
            ++cases;
            budget.configure(false, 8, 2, 20, true);
            advance(100, 100);
            check(budget.depth() == 8, "disabling adaptive mode restores full ceiling");
            ++cases;
            budget.configure(true, 8, 2, 5, false);
            advance(100, 100);
            check(budget.depth() == 8, "GPU-time mode ignores whole-game presentation timing");
            ++cases;
            budget.configure(true, 8, 2, 1000.0 / 144.0, true);
            uint64_t high_tick = 1000;
            for (unsigned iteration_index = 0; iteration_index < 1000; ++iteration_index) {
                high_tick += 4167;
                budget.present(8, high_tick, true);
            }
            check(budget.depth() == 8 && budget.ema() > 4 && budget.ema() < 4.3,
                  "sub-millisecond clock precision supports high-refresh presentation samples");
            ++cases;
        }
        {
            phoenix::DepthBudget budget;
            budget.configure(true, 8, 2, 5);
            for (unsigned iteration_index = 0; iteration_index < 29; ++iteration_index)
                check(!budget.sample(8, 10), "budget waits for fresh samples");
            check(budget.sample(8, 10) && budget.depth() == 7,
                  "budget lowers one block after 30 completed samples");
            ++cases;
            check(!budget.sample(8, 10) && budget.depth() == 7,
                  "old-depth asynchronous samples ignored");
            ++cases;
            check(!budget.sample(7, 0) &&
                      !budget.sample(7, std::numeric_limits<double>::infinity()),
                  "invalid GPU sample ignored");
            ++cases;
            for (unsigned iteration_index = 0; iteration_index < 180; ++iteration_index)
                budget.sample(budget.depth(), 10);
            check(budget.depth() == 2, "budget respects quality floor even when unattainable");
            ++cases;
            for (unsigned iteration_index = 0; iteration_index < 30; ++iteration_index)
                budget.sample(2, 1);
            check(budget.depth() == 3, "budget recovers depth after low cost");
            ++cases;
            budget.configure(false, 8, 2, 5);
            budget.sample(8, 100);
            check(budget.depth() == 8, "disabling budget restores ceiling");
            ++cases;
            budget.configure(true, 4, 1, 5);
            for (unsigned iteration_index = 0; iteration_index < 29; ++iteration_index)
                budget.sample(4, 10);
            budget.reset_samples();
            check(!budget.sample(4, 10) && budget.depth() == 4,
                  "timing reset discards budget window");
            ++cases;
        }
        bad = {};
        bad.divider = 0;
        check(!bad.valid(), "reject empty split band");
        ++cases;
        bad = {};
        bad.intensity = std::numeric_limits<float>::quiet_NaN();
        check(!bad.valid(), "reject nonfinite intensity");
        ++cases;
        bad = {};
        bad.controls.advanced = true;
        bad.controls.skin = .4f;
        bad.controls.auto_mask = true;
        bad.controls.motion_x = 2;
        bad.controls.depth = 0;
        auto guided = bad.controls.apply(Controls{});
        check(guided.auto_mask == 1 && guided.skin == .4f && guided.motion_x == 2 &&
                  guided.inverted == 0,
              "advanced guides wired");
        ++cases;
        bad.controls.motion_y = 5;
        check(!bad.valid(), "reject excessive motion scale");
        ++cases;
        bad.controls.motion_y = -4;
        bad.controls.motion_x = -4;
        check(bad.valid() && bad.controls.apply(Controls{}).motion_x == -4,
              "signed guide scales reach frame controls");
        ++cases;
        bad.controls.motion_y = -4.01f;
        check(!bad.valid(), "negative motion scale bound");
        ++cases;
        PhoenixLiveControls live;
        Controls game;
        game.tone = .7f;
        game.structure = .8f;
        game.reset = 1;
        check(live.apply(game).tone == .7f && live.apply(game).reset == 1,
              "live defaults preserve host");
        ++cases;
        live.enabled = true;
        live.tone = .25f;
        live.structure = .5f;
        auto applied = live.apply(game);
        check(applied.tone == .25f && applied.structure == .5f && game.tone == .7f,
              "live strengths reach controls without mutating host");
        ++cases;
        game.reset = 0;
        live.reset_requested = true;
        check(live.apply(game).reset == 1 && live.reset_requested,
              "reset survives until recording");
        ++cases;
        live.recorded();
        check(live.apply(game).reset == 0, "one-shot reset");
        ++cases;
        rejected([&] { effective_controls(game, true, false); });
        game.style = 2;
        game.intensity = .3f;
        game.ui_correction = 1;
        applied = live.apply(game);
        check(applied.style == 2 && applied.intensity == .3f && applied.ui_correction == 1,
              "live cannot hide unsupported host modes");
        ++cases;
        live.tone = std::numeric_limits<float>::quiet_NaN();
        rejected([&] { live.apply(game); });
        live.tone = std::numeric_limits<float>::infinity();
        rejected([&] { live.apply(game); });
        live.tone = -.01f;
        rejected([&] { live.apply(game); });
        live.tone = 2.01f;
        rejected([&] { live.apply(game); });
        live.tone = 0;
        live.structure = 0;
        check(live.valid(), "live zero endpoint");
        ++cases;
        live.tone = 1;
        live.structure = 1;
        check(live.valid(), "live unit endpoint");
        ++cases;
        live.structure = std::numeric_limits<float>::quiet_NaN();
        rejected([&] { live.apply(game); });
        live.structure = -1;
        rejected([&] { live.apply(game); });
        live.structure = 2.01f;
        rejected([&] { live.apply(game); });
        live.tone = 2;
        live.structure = 2;
        check(live.valid(), "two-times tone and structure endpoint");
        ++cases;
        phoenix::Settings reset_policy;
        reset_policy.reset_after_switch = 0;
        check(!reset_policy.valid(), "switch reset cannot be disabled");
        ++cases;
        reset_policy = {};
        reset_policy.reset_every = 601;
        check(!reset_policy.valid(), "reset cadence bound");
        ++cases;
        controls_tests();
        kernel_resolution_tests();
        ngx_shim_tests();
        sl_shim_tests();
        dispatch_log_tests();
        batch_tests();
        cubins_tests();
        skip_tests();
        panel_tests();
        observe_tests();
        routing_defaults_tests();
        Transaction valid;
        valid.begin(1, 2);
        check(valid.pending() && valid.committed() == 0, "record must not commit");
        ++cases;
        check(valid.submission_matches(1, 2) && !valid.submission_matches(1, 3) &&
                  !valid.submission_matches(3, 2),
              "submission must be validated before Execute");
        ++cases;
        rejected([&] { valid.begin(3, 2); });
        valid.submitted(1, 2, true);
        check(!valid.pending() && valid.committed() == 1, "submit commit");
        ++cases;
        valid.begin(3, 2);
        valid.submitted(3, 2, true);
        ++cases;
        rejected([&] { valid.begin(1, 4); });
        Transaction wrong;
        wrong.begin(1, 2);
        rejected([&] { wrong.submitted(9, 2, true); });
        check(wrong.failed() && wrong.committed() == 0, "wrong list must poison");
        ++cases;
        Transaction wrong_queue;
        wrong_queue.begin(1, 2);
        rejected([&] { wrong_queue.submitted(1, 3, true); });
        Transaction failure;
        failure.begin(1, 2);
        rejected([&] { failure.submitted(1, 2, false); });
        rejected([&] { failure.begin(1, 2); });
        Transaction abandoned;
        abandoned.begin(1, 2);
        abandoned.invalidate();
        rejected([&] { abandoned.submitted(1, 2, true); });
        rejected([&] { abandoned.begin(1, 2); });
        Transaction duplicate;
        rejected([&] { duplicate.submitted(1, 2, true); });
        // A measured Unreal D3D12 frame records the boundary on one thread and
        // submits on another. Identity, not thread affinity, authorises the
        // commit; the caller serialises the two with its own lock.
        Transaction other_thread;
        other_thread.begin(1, 2);
        bool accepted = false, prevalidated = false;
        std::thread worker([&] {
            prevalidated = other_thread.submission_matches(1, 2);
            try {
                other_thread.submitted(1, 2, true);
                accepted = true;
            } catch (...) {
            }
        });
        worker.join();
        check(prevalidated && accepted && !other_thread.failed() && other_thread.committed() == 1 &&
                  other_thread.cross_thread_submission(),
              "cross-thread submission with matching identity commits");
        ++cases;
        check(other_thread.owner_thread(), "recording ownership stays with the recording thread");
        ++cases;
        Transaction wrong_identity;
        wrong_identity.begin(1, 2);
        bool rejected_elsewhere = false;
        std::thread other([&] {
            if (wrong_identity.submission_matches(9, 2))
                return;
            try {
                wrong_identity.submitted(9, 2, true);
            } catch (...) {
                rejected_elsewhere = true;
            }
        });
        other.join();
        check(rejected_elsewhere && wrong_identity.failed(),
              "cross-thread submission of a foreign list is refused");
        ++cases;
        Transaction foreign_record;
        bool recording_refused = false;
        std::thread recorder([&] {
            try {
                foreign_record.begin(1, 2);
            } catch (...) {
                recording_refused = true;
            }
        });
        recorder.join();
        check(recording_refused, "recording from another thread is still refused");
        ++cases;
        // The thread-change session boundary branches on exactly this: an
        // idle (submitted, settlable) transaction still reports a foreign
        // thread as non-owner, so prepare_automatic retires and rebuilds
        // instead of recording foreign.
        Transaction migrated;
        migrated.begin(1, 2);
        migrated.submitted(1, 2, true);
        bool foreign_owner = true;
        std::thread probe([&] { foreign_owner = migrated.owner_thread(); });
        probe.join();
        check(!foreign_owner && !migrated.failed(),
              "idle transaction still names a foreign thread non-owner");
        ++cases;
        ledger_tests();
        manifest_tests();
        std::cout << "{\"cpu_only\":true,\"passed\":true,\"cases\":" << cases << "}\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
