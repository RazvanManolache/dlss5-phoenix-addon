#pragma once
#include "../../pipeline/options.h"
#include "../../../third_party/nlohmann/json.hpp"
#include <limits>

namespace phoenix_compat {
// The helper owns these GPU services; they are separate from its core's live
// artistic controls and do not require a model/process restart when edited.
template<class Visitor> void visit_helper_motion(phoenix_pipeline::Options& p, Visitor visit) {
    visit("MotionEstimation",p.motion_enabled);
    visit("MotionProvider",p.motion_source);
    visit("FlowAnalysisHeight",p.flow_height);
    visit("FlowConsistencyPixels",p.flow_consistency);
    visit("FlowCostThreshold",p.flow_cost);
    visit("MotionDiagnosticView",p.motion_view);
    visit("SceneDetection",p.scene_detection);
}
inline bool copy_helper_motion(phoenix_pipeline::Options& target,const phoenix_pipeline::Options& source) {
    auto fields=[](auto& p){return std::tie(p.motion_enabled,p.motion_source,p.flow_height,
        p.flow_consistency,p.flow_cost,p.motion_view,p.scene_detection);};
    const bool changed=fields(target)!=fields(source);fields(target)=fields(source);return changed;
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
inline bool helper_history_reset(bool explicit_reset,bool first_frame,bool motion_valid,bool discontinuity) {
    return explicit_reset||first_frame||!motion_valid||discontinuity;
}
} // namespace phoenix_compat
