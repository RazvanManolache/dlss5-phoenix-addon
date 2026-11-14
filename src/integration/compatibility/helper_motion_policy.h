#pragma once
#include "../../pipeline/options.h"
#include "../../../third_party/nlohmann/json.hpp"
#include <cmath>
#include <limits>
#include <span>
#if defined(_M_IX86) || defined(_M_X64)
#include <emmintrin.h>
#endif

namespace phoenix_compat {
// A failed explicit game-depth capture never enables an estimator, including
// in Depth only. Only Auto may choose a different source.
inline int legacy_depth_estimator_selection(int source) {
    return source == 4 ? 3 : source;
}

// The helper owns these GPU services; they are separate from its core's live
// artistic controls and do not require a model/process restart when edited.
template<class Visitor> void visit_helper_motion(phoenix_pipeline::Options& p, Visitor visit) {
    phoenix_pipeline::visit_depth_options(p, visit);
    visit("MotionEstimation",p.motion_enabled);
    visit("MotionProvider",p.motion_source);
    visit("FlowAnalysisHeight",p.flow_height);
    visit("FlowConsistencyPixels",p.flow_consistency);
    visit("FlowCostThreshold",p.flow_cost);
    visit("MotionDiagnosticView",p.motion_view);
    visit("SceneDetection",p.scene_detection);
}
template<class Options> auto legacy_depth_fields(Options& p) {
    return std::tie(p.legacy_depth, p.legacy_depth_strength, p.legacy_depth_preview, p.legacy_depth_invert,
                    p.legacy_depth_fixed_range, p.legacy_depth_near, p.legacy_depth_far);
}
inline bool copy_helper_motion(phoenix_pipeline::Options& target,const phoenix_pipeline::Options& source) {
    auto fields=[](auto& p){return std::tie(p.motion_enabled,p.motion_source,p.flow_height,
        p.flow_consistency,p.flow_cost,p.motion_view,p.scene_detection);};
    const bool changed=fields(target)!=fields(source) || legacy_depth_fields(target)!=legacy_depth_fields(source);
    fields(target)=fields(source);
    legacy_depth_fields(target)=legacy_depth_fields(source);
    return changed;
}
inline nlohmann::json helper_motion_json(phoenix_pipeline::Options p) {
    auto result=nlohmann::json::object();visit_helper_motion(p,[&](const char* key,auto value){result[key]=value;});return result;
}
// Transactional: leave both caller objects unchanged on a malformed patch.
// Remaining keys belong to the core and retain its existing strict validation.
inline void split_helper_motion_patch(phoenix_pipeline::Options& options,nlohmann::json& core_patch) {
    if(!core_patch.is_object())throw std::invalid_argument("Motion controls must be an object");
    auto next=options;auto remaining=core_patch;
    visit_helper_motion(next,[&](const char* key,auto& field){
        if(auto found=remaining.find(key);found!=remaining.end()){
            using T=std::decay_t<decltype(field)>;
            if constexpr(std::is_same_v<T,bool>){if(!found->is_boolean())throw std::invalid_argument("Expected motion checkbox value");}
            else if(!found->is_number() || (std::is_integral_v<T> && !found->is_number_integer()))
                throw std::invalid_argument("Expected numeric motion setting");
            if constexpr(std::is_integral_v<T> && !std::is_same_v<T,bool>){
                const auto number=found->template get<double>();
                if(number<std::numeric_limits<T>::lowest()||number>std::numeric_limits<T>::max())
                    throw std::invalid_argument("Motion integer overflow");
            }
            field=found->template get<T>();remaining.erase(found);
        }
    });
    if(!next.valid())throw std::invalid_argument("Motion controls outside supported limits");
    options=next;core_patch=std::move(remaining);
}
// One sweep replacing the separate finite/range validation and min/max
// passes callers used to run over every depth plane.
struct DepthRange {
    bool valid = false;
    float minimum = 0.f, maximum = 0.f;
};
inline DepthRange depth_range_stats(std::span<const float> depth) {
    if (depth.empty()) return {};
    DepthRange result{true, std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()};
#if defined(_M_IX86) || defined(_M_X64)
    const auto zero = _mm_setzero_ps(), one = _mm_set1_ps(1.f);
    auto minimum = _mm_set1_ps(result.minimum), maximum = _mm_set1_ps(result.maximum);
    auto valid = _mm_castsi128_ps(_mm_set1_epi32(-1));
    size_t i = 0;
    for (; i + 4 <= depth.size(); i += 4) {
        const auto values = _mm_loadu_ps(depth.data() + i);
        const auto in_range = _mm_and_ps(_mm_cmpge_ps(values, zero), _mm_cmple_ps(values, one));
        valid = _mm_and_ps(valid, in_range);
        // MINPS/MAXPS select the second operand if the sample is NaN.
        minimum = _mm_min_ps(values, minimum);
        maximum = _mm_max_ps(values, maximum);
    }
    if (i) {
        alignas(16) float low[4], high[4];
        _mm_store_ps(low, minimum);
        _mm_store_ps(high, maximum);
        result.minimum = *(std::min_element)(low, low + 4);
        result.maximum = *(std::max_element)(high, high + 4);
        result.valid = _mm_movemask_ps(valid) == 15;
    }
    depth = depth.subspan(i);
#endif
    for (const float value : depth) {
        result.valid &= std::isfinite(value) && value >= 0.f && value <= 1.f;
        result.minimum = (std::min)(result.minimum, value);
        result.maximum = (std::max)(result.maximum, value);
    }
    return result;
}

// Input is conventional finite depth in [0,1]. Pivot at the center of the
// observed range so hardware Z values clustered near 1 do not all clip.
inline float adjust_depth_value(float value, float center, float strength) {
    const float delta = value - center;
    if (strength <= 1.f) return center + strength * delta;
    if (delta == 0.f) return value;
    const float room = delta > 0.f ? 1.f - center : center;
    const float adjustment = strength / (1.f + (strength - 1.f) * (std::abs(delta) / room));
    return std::clamp(center + adjustment * delta, 0.f, 1.f);
}
inline void adjust_legacy_depth(std::span<float> depth, float strength) {
    if (depth.empty() || strength == 1.f) return;
    const auto [minimum, maximum] = std::minmax_element(depth.begin(), depth.end());
    const float center = (*minimum + *maximum) * .5f;
    for (auto& value : depth) value = adjust_depth_value(value, center, strength);
}

// Display near values as white using the source range before strength tuning.
inline void remap_legacy_depth(std::span<float> depth, float near_depth, float far_depth) {
    const float scale = 1.f / (far_depth - near_depth);
    for (auto& value : depth)
        value = std::clamp((value - near_depth) * scale, 0.f, 1.f);
}

// Range controls refer to the observed span of this source frame. Device Z
// often occupies a narrow band near 1. Strength operates on that band's
// normalized coordinates so its full contrast range is available.
inline bool prepare_legacy_depth(std::span<float> depth, float minimum, float maximum,
                                 const phoenix_pipeline::Options& options, bool preview) {
    if (depth.empty() || maximum - minimum <= 1e-6f) return false;
    const bool windowed = options.legacy_depth_fixed_range &&
        (options.legacy_depth_near > 0.f || options.legacy_depth_far < 1.f);
    if (!preview && !windowed && options.legacy_depth_strength == 1.f)
        return true;

    // Fused: both remaps in one sweep while tracking the remapped range,
    // then strength adjustment and the return to native Z in a second sweep.
    // This replaces the remap + min/max + adjust + lerp sequence value for
    // value; the tracked range is the range adjust used to recompute.
    const float scale = 1.f / (maximum - minimum);
    const float window_scale = windowed ? 1.f / (options.legacy_depth_far - options.legacy_depth_near) : 1.f;
    float remapped_minimum = std::numeric_limits<float>::infinity();
    float remapped_maximum = -std::numeric_limits<float>::infinity();
    for (auto& value : depth) {
        value = std::clamp((value - minimum) * scale, 0.f, 1.f);
        if (windowed)
            value = std::clamp((value - options.legacy_depth_near) * window_scale, 0.f, 1.f);
        remapped_minimum = (std::min)(remapped_minimum, value);
        remapped_maximum = (std::max)(remapped_maximum, value);
    }
    const float center = (remapped_minimum + remapped_maximum) * .5f;
    const float strength = options.legacy_depth_strength;
    for (auto& value : depth) {
        if (strength != 1.f) value = adjust_depth_value(value, center, strength);
        // Preserve native Z units for the model guide. The preview
        // displays the adjusted normalized map without stretching it again.
        if (!preview) value = std::lerp(minimum, maximum, value);
    }
    return true;
}

inline void preview_legacy_depth(std::span<const float> depth, float minimum, float maximum,
                                 std::span<uint8_t> rgba, bool invert = false) {
    const float range = maximum - minimum;
    for (size_t i = 0; i < rgba.size() / 4; ++i) {
        float proximity = 0.f;
        if (i < depth.size() && range > 1e-6f) {
            const float distance = std::clamp((depth[i] - minimum) / range, 0.f, 1.f);
            proximity = invert ? distance : 1.f - distance;
        }
        const auto gray = static_cast<uint8_t>(std::lround(proximity * 255.f));
        rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = gray;
        rgba[i * 4 + 3] = 255;
    }
}

inline bool helper_history_reset(bool explicit_reset,bool first_frame,bool motion_valid,bool discontinuity) {
    return explicit_reset||first_frame||!motion_valid||discontinuity;
}
} // namespace phoenix_compat
