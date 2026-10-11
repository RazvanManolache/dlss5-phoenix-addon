#pragma once
// Streamline tag-harvest tables (renodx replacement, sl stage). Pure logic:
// the addon.cpp sl detours translate sl structs to these primitives, so
// this header stays free of sl/d3d includes and CPU-tests hermetically.
// Tags arrive per (frame, viewport); evaluates join by frame index. The
// template assumes a single viewport (menu/fullscreen); multi-viewport
// frames merge with the viewport count recorded, never silently.
// NOT thread-safe: every touch serializes on the addon lock.
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>
#include "ngx_shim.h"

namespace phoenix_addon {
namespace sl_shim {
// sl::Feature values we template on (sl_core_types.h).
inline constexpr uint64_t kFeatureDLSSRR = 1001;
// sl::BufferType values we read (sl_core_types.h).
inline constexpr unsigned kTagDepth = 0;
inline constexpr unsigned kTagMotion = 1;
inline constexpr unsigned kTagHUDLessColor = 2;
inline constexpr unsigned kTagScalingInputColor = 3;
// Telemetry geometry uses the existing compact packet spelling: w/h are
// resource width/height; ex_w/ex_h are the tagged viewport extent. These are
// observations, not allocated frame buffers.
struct SlTag {
    uint64_t resource_key = 0;
    unsigned buffer_type = 0;
    unsigned w = 0, h = 0, format = 0;
    unsigned ex_w = 0, ex_h = 0;
};
// Effective geometry: Resource dims when the game fills them, else the
// tag extent (games that leave Resource at zero still scope the area).
inline void sl_tag_geometry(SlTag& tag) {
    if ((!tag.w || !tag.h) && tag.ex_w && tag.ex_h) {
        tag.w = tag.ex_w;
        tag.h = tag.ex_h;
    }
}
// One joined-table row for forensics.
struct SlTypeRow {
    unsigned type = 0, w = 0, h = 0, format = 0, ex_w = 0, ex_h = 0;
};
struct SlInputs {
    uint64_t feature = 0;
    unsigned color_w = 0, color_h = 0, color_format = 0;
    bool motion = false, depth = false;
    unsigned out_w = 0, out_h = 0;
    unsigned viewports = 0;
    uint64_t tags = 0, evaluates = 0;
    uint32_t last_frame = 0;
};
class SlObserver {
  public:
    void note_tags(uint32_t frame, const SlTag* tags, size_t tag_count, unsigned viewport);
    // Deprecated frame-less slSetTag path: applies to every frame until
    // replaced. Joined only when no same-frame tags exist.
    void note_tags_global(const SlTag* tags, size_t tag_count);
    void note_evaluate(uint64_t feature, uint32_t frame, unsigned out_w, unsigned out_h);
    uint32_t last_tag_frame() const {
        return last_tag_frame_;
    }
    size_t global_tags() const {
        return global_.size();
    }
    const std::map<uint64_t, SlInputs>& features() const {
        return features_;
    }
    // Newest-frame live pointers for piggyback driving (identity keys; the
    // caller reinterprets them the same frame they were tagged, never
    // across frames). Color prefers scaling-input, falls back to HUD-less;
    // motion is required (the claim path reads DLSSNR.MVec unconditionally).
    bool current_inputs(uint64_t& color, uint64_t& motion, uint64_t& depth) const {
        color = motion = depth = 0;
        if (frames_.empty())
            return false;
        const auto& by_type = frames_.rbegin()->second.by_type;
        const auto sit = by_type.find(kTagScalingInputColor);
        const auto hit = (sit == by_type.end()) ? by_type.find(kTagHUDLessColor) : sit;
        if (hit == by_type.end() || !hit->second.w)
            return false;
        const auto mit = by_type.find(kTagMotion);
        if (mit == by_type.end() || !mit->second.resource_key)
            return false;
        color = hit->second.resource_key;
        motion = mit->second.resource_key;
        const auto dit = by_type.find(kTagDepth);
        if (dit != by_type.end())
            depth = dit->second.resource_key;
        return true;
    }
    // Newest joined table rows (type/geometry as tagged, no inference).
    std::vector<SlTypeRow> last_table() const {
        std::vector<SlTypeRow> rows;
        if (frames_.empty())
            return rows;
        for (const auto& entry : frames_.rbegin()->second.by_type) {
            const SlTag& tag = entry.second;
            SlTypeRow row;
            row.type = tag.buffer_type;
            row.w = tag.w;
            row.h = tag.h;
            row.format = tag.format;
            row.ex_w = tag.ex_w;
            row.ex_h = tag.ex_h;
            rows.push_back(row);
        }
        return rows;
    }
    const SlInputs* find(uint64_t feature) const {
        const auto lookup_iterator = features_.find(feature);
        return lookup_iterator == features_.end() ? nullptr : &lookup_iterator->second;
    }
    // Latest-evaluated RR feature with real color geometry.
    const SlInputs* drive_template() const {
        const SlInputs* best = nullptr;
        for (const auto& entry : features_) {
            const SlInputs& inputs = entry.second;
            if (inputs.feature == kFeatureDLSSRR && inputs.color_w != 0 && inputs.color_h != 0 &&
                (!best || inputs.evaluates > best->evaluates))
                best = &inputs;
        }
        return best;
    }
    size_t size() const {
        return features_.size();
    }
    size_t frames_held() const {
        return frames_.size();
    }

  private:
    struct FrameTags {
        uint32_t frame = 0;
        uint32_t viewport_mask = 0;
        std::map<unsigned, SlTag> by_type;
    };
    static unsigned popcount(uint32_t bits) {
        unsigned bit_count = 0;
        while (bits) {
            bit_count += bits & 1u;
            bits >>= 1;
        }
        return bit_count;
    }
    std::map<uint32_t, FrameTags> frames_;
    std::map<unsigned, SlTag> global_;
    std::map<uint64_t, SlInputs> features_;
    uint32_t last_tag_frame_ = 0;
};
inline void SlObserver::note_tags_global(const SlTag* tags, size_t tag_count) {
    if (!tags || !tag_count)
        return;
    for (size_t iteration_index = 0; iteration_index < tag_count; ++iteration_index) {
        if (!tags[iteration_index].resource_key)
            continue;
        SlTag tag = tags[iteration_index];
        sl_tag_geometry(tag);
        global_[tag.buffer_type] = tag;
    }
}
inline void SlObserver::note_tags(uint32_t frame, const SlTag* tags, size_t tag_count,
                                  unsigned viewport) {
    if (!tags || !tag_count)
        return;
    FrameTags& frame_tags = frames_[frame];
    frame_tags.frame = frame;
    last_tag_frame_ = frame;
    if (viewport < 32)
        frame_tags.viewport_mask |= (1u << viewport);
    for (size_t iteration_index = 0; iteration_index < tag_count; ++iteration_index) {
        if (!tags[iteration_index].resource_key)
            continue;
        SlTag tag = tags[iteration_index];
        sl_tag_geometry(tag);
        frame_tags.by_type[tag.buffer_type] = tag;
    }
    while (frames_.size() > 8)
        frames_.erase(frames_.begin());
}
inline void SlObserver::note_evaluate(uint64_t feature, uint32_t frame, unsigned out_w,
                                      unsigned out_h) {
    if (!feature)
        return;
    SlInputs& inputs = features_[feature];
    inputs.feature = feature;
    ++inputs.evaluates;
    inputs.last_frame = frame;
    if (out_w && out_h) {
        inputs.out_w = out_w;
        inputs.out_h = out_h;
    }
    const auto fit = frames_.find(frame);
    const std::map<unsigned, SlTag>& table = (fit != frames_.end()) ? fit->second.by_type : global_;
    if (fit != frames_.end())
        inputs.viewports = popcount(fit->second.viewport_mask);
    // Scaling input color is the render-resolution color; HUD-less is the
    // fallback. Either is real geometry, never a guess.
    const SlTag* color = nullptr;
    const auto sit = table.find(kTagScalingInputColor);
    if (sit != table.end())
        color = &sit->second;
    else {
        const auto hit = table.find(kTagHUDLessColor);
        if (hit != table.end())
            color = &hit->second;
    }
    if (color && color->w && color->h) {
        inputs.color_w = color->w;
        inputs.color_h = color->h;
        inputs.color_format = color->format;
    }
    inputs.motion = table.count(kTagMotion) != 0;
    inputs.depth = table.count(kTagDepth) != 0;
    uint64_t tag_count = 0;
    for (const auto& entry : table)
        tag_count += entry.second.resource_key ? 1 : 0;
    inputs.tags = tag_count;
}
struct SlDriveSizes {
    unsigned in_w = 0, in_h = 0, format = 0;
    bool valid = false;
};
// A self-driven NR feature consumes the game's render-resolution color
// geometry exactly as tagged. Anything missing refuses.
inline SlDriveSizes negotiate_sl_sizes(const SlInputs* tpl) {
    SlDriveSizes sizes;
    if (!tpl)
        return sizes;
    sizes.in_w = tpl->color_w;
    sizes.in_h = tpl->color_h;
    sizes.format = tpl->color_format;
    sizes.valid = sizes.in_w != 0 && sizes.in_h != 0;
    if (!sizes.valid)
        sizes = SlDriveSizes{};
    return sizes;
}
// Self-drive NGX attempt order, single-sourced from the NGX ids:
// renodx parity first, public RR fallback. 0 means exhausted.
inline unsigned selfdrive_try_id(unsigned attempt) {
    return attempt < 3 ? ngx_shim::kSelfDriveTryIds[attempt] : 0;
}
} // namespace sl_shim
} // namespace production_game41
