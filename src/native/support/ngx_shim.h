#pragma once
// NGX export-observation tables + self-drive size negotiation (renodx
// replacement, stages A/B). Pure logic: no GPU, no hooks, no logging. The
// addon.cpp export detours record here; the self-drive path reads here.
// CPU-tested in tests/native/tests.cpp (ngx_shim_tests()).
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

namespace phoenix_addon {
namespace ngx_shim {
// NVSDK_NGX_Feature enum value for ray reconstruction / DLSS denoise. The
// game creates this itself (observed: "feature=13 (DLSSD/RR)"), so it is
// the template a self-driven feature mirrors.
inline constexpr unsigned kRayReconstruction = 13;
inline constexpr unsigned kSuperSampling = 1;
// NGX IDs only. Streamline has a separate feature-ID namespace.
inline constexpr bool is_direct_game_feature(unsigned feature_id) {
    return feature_id == 0 || feature_id == kSuperSampling ||
           feature_id == kRayReconstruction;
}
struct Observation {
    uint64_t key = 0;
    unsigned feature_id = 0;
    unsigned in_width = 0, in_height = 0, out_width = 0, out_height = 0;
    std::string creator;
    uint64_t creates = 0, evaluates = 0, releases = 0;
    unsigned create_flags = 0;
};
// NOT thread-safe by itself: every Observer touch (including find/)
// serializes on the addon lock, because export detours run on engine task
// threads. An unlocked table wedged Bonsai past its deadline.
class Observer {
  public:
    void note_create(uint64_t key, unsigned feature_id, unsigned in_w, unsigned in_h,
                     unsigned out_w, unsigned out_h, const char* creator,
                     unsigned create_flags = 0) {
        if (!key)
            return;
        Observation& observation = features_[key];
        observation.key = key;
        observation.feature_id = feature_id;
        observation.create_flags = create_flags;
        observation.in_width = in_w;
        observation.in_height = in_h;
        observation.out_width = out_w;
        observation.out_height = out_h;
        if (creator && *creator)
            observation.creator = creator;
        ++observation.creates;
    }
    void note_evaluate(uint64_t key) {
        if (!key)
            return;
        Observation& observation = features_[key];
        observation.key = key;
        ++observation.evaluates;
    }
    void note_release(uint64_t key) {
        features_.erase(key);
    }
    const Observation* find(uint64_t key) const {
        const auto lookup_iterator = features_.find(key);
        return lookup_iterator == features_.end() ? nullptr : &lookup_iterator->second;
    }
    // First game-side ray-reconstruction feature with observed evaluations.
    // A renodx-style foreign feature evaluates too, but creator names the
    // author, so the self-drive stage can require an engine creator.
    const Observation* drive_template() const {
        for (const auto& entry : features_) {
            const Observation& observation = entry.second;
            if (observation.feature_id == kRayReconstruction && observation.evaluates > 0)
                return &observation;
        }
        return nullptr;
    }
    size_t size() const {
        return features_.size();
    }

  private:
    std::map<uint64_t, Observation> features_;
};
struct SelfDriveSizes {
    unsigned in_w = 0, in_h = 0, out_w = 0, out_h = 0;
    bool valid = false;
};
// A self-driven feature mirrors the template exactly (renodx parity: same
// NR input/output + guides as the observed game feature). Anything missing
// refuses instead of guessing a resolution.
inline SelfDriveSizes negotiate_self_drive(const Observation* tpl) {
    SelfDriveSizes sizes;
    if (!tpl)
        return sizes;
    sizes.in_w = tpl->in_width;
    sizes.in_h = tpl->in_height;
    sizes.out_w = tpl->out_width;
    sizes.out_h = tpl->out_height;
    sizes.valid = sizes.in_w != 0 && sizes.in_h != 0 && sizes.out_w != 0 && sizes.out_h != 0;
    if (!sizes.valid)
        sizes = SelfDriveSizes{};
    return sizes;
}
// Self-drive arming (renodx replacement, stage C): engage only when a game
// template exists and no foreign driver is present. renodx presence is
// observed by module, never by creator string (forwarded exports misname
// the direct caller, so creator is forensic info only).
inline bool self_drive_armed(const SelfDriveSizes& sizes, bool renodx_present) {
    return sizes.valid && !renodx_present;
}
// Try SuperSampling first: the proven DLSS5-NR vehicle (public MIT
// DLSS5-Feeder: create SuperSampling+DLAA on its own device, then switch NR
// on at evaluate with DLSSNR.Enabled). Then the renodx-parity id-13 RR the
// game itself uses, then ordinal 18 last (0xBAD0000B through slot 0).
inline constexpr unsigned kSelfDriveTryIds[] = {0u, 13u, 18u};
// Retry-cooldown gate (GetTickCount64 ms domain): transient self-drive
// failures (unattributed queue, missing inputs) wait out a cooldown and try
// again instead of latching failed. Definitive failures (NGX entries absent,
// create exhausted) latch failed at their own sites.
inline constexpr unsigned long long kSelfDriveRetryCooldownMs = 5000;
inline bool selfdrive_retry_due(unsigned long long now_tick, unsigned long long retry_after_tick) {
    return now_tick >= retry_after_tick;
}
} // namespace ngx_shim
} // namespace production_game41
