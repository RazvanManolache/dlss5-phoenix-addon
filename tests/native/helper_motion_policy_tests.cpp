// CPU-only: source priority, transactional controls, history and guide lifetime.
#include "../../src/integration/compatibility/helper_motion_policy.h"
#include "../../src/integration/compatibility/protocol.h"
#include "../../src/integration/compatibility/session_policy.h"
#include <iostream>

int main() try {
    unsigned checks=0;
    auto check=[&](bool ok){++checks;if(!ok)throw std::runtime_error("Motion policy check "+std::to_string(checks));};
    using namespace phoenix_pipeline;
    using namespace phoenix_compat;
    Options p;
    for(bool estimate:{false,true})for(int source=0;source<4;++source)
        for(bool engine:{false,true})for(bool hardware:{false,true}){
            p.motion_enabled=estimate;p.motion_source=source;
            const auto result=choose_motion(p,engine,hardware).source;
            const auto expected=source==3?MotionSource::Zero:
                engine&&(source==0||source==1)?MotionSource::Engine:
                estimate&&hardware&&(source==0||source==2)?MotionSource::Nvidia:MotionSource::Zero;
            check(result==expected);
        }
    for(bool explicit_reset:{false,true})for(bool first:{false,true})
        for(bool valid:{false,true})for(bool cut:{false,true})
            check(helper_history_reset(explicit_reset,first,valid,cut)==(explicit_reset||first||!valid||cut));
    for(unsigned height:{64u,180u,360u,720u,1080u,2160u}){
        const auto extent=flow_extent(1920,1080,height);
        check(extent[0]<=1920&&extent[1]<=1080&&extent[0]>0&&extent[1]>0);
    }
    check(flow_extent(1920,1080,180)==std::array<unsigned,2>{320,180});
    check(flow_extent(2560,1080,360)==std::array<unsigned,2>{853,360});
    p={};auto patch=nlohmann::json{{"MotionEstimation",true},{"MotionProvider",2},
        {"FlowAnalysisHeight",2160},{"FlowConsistencyPixels",2.5},{"FlowCostThreshold",200},
        {"MotionDiagnosticView",2},{"SceneDetection",true},{"ForceReset",false}};
    split_helper_motion_patch(p,patch);
    check(p.motion_enabled&&p.motion_source==2&&p.flow_height==2160&&p.scene_detection);
    check(patch==nlohmann::json{{"ForceReset",false}});
    const auto before=helper_motion_json(p);
    for(auto invalid:std::vector<nlohmann::json>{
        {{"MotionEstimation",1}},{{"MotionProvider",4}},{{"MotionProvider",1.5}},
        {{"MotionProvider",uint64_t(1)<<32}},
        {{"FlowAnalysisHeight",63}},{{"FlowAnalysisHeight",2161}},
        {{"FlowConsistencyPixels",0}},{{"FlowCostThreshold",1001}},
        {{"MotionDiagnosticView",3}},{{"SceneDetection","true"}}}){
        const auto original=invalid;bool refused=false;
        try{split_helper_motion_patch(p,invalid);}catch(const std::exception&){refused=true;}
        check(refused&&helper_motion_json(p)==before&&invalid==original);
    }
    FeedGuideState pending{1,1920.f,-1080.f};check(feed_guides_valid(pending));
    check(feed_take_guides(pending).engine_motion==1);check(!feed_take_guides(pending).engine_motion);
    check(!feed_guides_valid({1,std::numeric_limits<float>::quiet_NaN(),1}));
    check(!feed_guides_valid({2,1,1}));
    check(feed_protocol_compatible(10)&&feed_protocol_compatible(11)&&feed_protocol_compatible(12)&&!feed_protocol_compatible(13));
    phoenix_addon::phoenix::Settings settings;
    apply_live_tuning_patch(settings,{{"ForceReset",true},{"ResetAfterSwitch",3},{"ResetEvery",7}});
    check(settings.force_reset&&settings.reset_after_switch==3&&settings.reset_every==7);
    check(live_tuning_json(settings).at("ForceReset")==true);
    std::cout<<nlohmann::json{{"passed",true},{"checks",checks},{"bits",sizeof(void*)*8}}.dump()<<'\n';
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
