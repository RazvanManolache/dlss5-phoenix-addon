// CPU-only: source priority, transactional controls, history and guide lifetime.
#include "../../src/integration/compatibility/helper_motion_policy.h"
#include "../../src/integration/compatibility/depth_freshness.h"
#include "../../src/integration/compatibility/protocol.h"
#include "../../src/integration/compatibility/session_policy.h"
#include <iostream>

int main() try {
    unsigned checks=0;
    auto check=[&](bool ok){++checks;if(!ok)throw std::runtime_error("Motion policy check "+std::to_string(checks));};
    using namespace phoenix_pipeline;
    using namespace phoenix_compat;
    for (int source = 0; source <= 4; ++source)
        check(legacy_depth_estimator_selection(source) == (source == 4 ? 3 : source));
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
    p = {};
    check(choose_motion(p, true, true, true).source == MotionSource::Engine);
    check(choose_motion(p, false, true, true).source == MotionSource::FidelityFX);
    check(choose_motion(p, false, false, true).source == MotionSource::FidelityFX);
    check(choose_motion(p, false, true, false).source == MotionSource::Nvidia);
    check(choose_motion(p, false, false, false).source == MotionSource::Zero);
    p.motion_enabled = false;
    check(choose_motion(p, true, true, true).source == MotionSource::Engine);
    check(choose_motion(p, false, true, true).source == MotionSource::Zero);
    p.motion_enabled = true;
    p.motion_source = int(MotionSource::Nvidia);
    check(choose_motion(p, true, true, true).source == MotionSource::Nvidia);
    p.motion_source = int(MotionSource::DIS);
    check(choose_motion(p, true, true, true).source == MotionSource::DIS);
    check(choose_motion(p, true, true, false).source == MotionSource::Zero);
    p.motion_source = int(MotionSource::Zero);
    check(choose_motion(p, true, true, true).source == MotionSource::Zero);
    for(bool explicit_reset:{false,true})for(bool first:{false,true})
        for(bool valid:{false,true})for(bool cut:{false,true})
            check(helper_history_reset(explicit_reset,first,valid,cut)==(explicit_reset||first||!valid||cut));
    for(unsigned height:{64u,180u,360u,720u,1080u,2160u}){
        const auto extent=flow_extent(1920,1080,height);
        check(extent[0]<=1920&&extent[1]<=1080&&extent[0]>0&&extent[1]>0);
    }
    check(flow_extent(1920,1080,180)==std::array<unsigned,2>{320,180});
    check(flow_extent(2560,1080,360)==std::array<unsigned,2>{853,360});
    for (auto source : {MotionSource::FidelityFX, MotionSource::DIS, MotionSource::Video, MotionSource::BlockMatch}) {
        Options selected{};
        nlohmann::json provider_patch{{"MotionProvider", int(source)}};
        split_helper_motion_patch(selected, provider_patch);
        check(provider_patch.empty());
        check(selected.motion_source == int(source) && selected.valid());
    }
    p={};check(p.legacy_depth==0);auto patch=nlohmann::json{{"MotionEstimation",true},{"MotionProvider",2},
        {"FlowAnalysisHeight",2160},{"FlowConsistencyPixels",2.5},{"FlowCostThreshold",200},
        {"MotionDiagnosticView",2},{"SceneDetection",true},{"LegacyDepth",4},{"ForceReset",false}};
    split_helper_motion_patch(p,patch);
    check(p.motion_enabled&&p.motion_source==2&&p.flow_height==2160&&p.scene_detection&&p.legacy_depth==4);
    check(patch==nlohmann::json{{"ForceReset",false}});
    auto depth_patch = nlohmann::json{{"LegacyDepthStrength", .501}, {"LegacyDepthPreview", true},
        {"LegacyDepthInvert", true}, {"LegacyDepthFixedRange", true}, {"LegacyDepthNear", .90}, {"LegacyDepthFar", .96}};
    split_helper_motion_patch(p, depth_patch);
    check(depth_patch.empty() && p.legacy_depth_strength == .501f && p.legacy_depth_preview &&
          p.legacy_depth_invert && p.legacy_depth_fixed_range && p.legacy_depth_near == .90f && p.legacy_depth_far == .96f);
    Options copied;
    check(copy_helper_motion(copied, p) && helper_motion_json(copied) == helper_motion_json(p));
    auto max_strength = nlohmann::json{{"LegacyDepthStrength", 10.f}};
    split_helper_motion_patch(p, max_strength);
    check(max_strength.empty() && p.legacy_depth_strength == 10.f);
    const std::array depth{.89f, .94f, .98f};
    const std::array samples{.5f, .3f, .7f, .4f, .9f, .1f, .8f, .2f, 1.f};
    auto range = depth_range_stats(samples);
    check(range.valid && range.minimum == .1f && range.maximum == 1.f);
    auto bad_samples = samples;
    bad_samples[1] = std::numeric_limits<float>::quiet_NaN();
    range = depth_range_stats(bad_samples);
    check(!range.valid && range.minimum == .1f && range.maximum == 1.f);
    bad_samples = samples;
    bad_samples.back() = 1.1f;
    check(!depth_range_stats(bad_samples).valid);
    check(!depth_range_stats({}).valid);
    auto reduced = depth, amplified = depth, unchanged = depth;
    adjust_legacy_depth(reduced, .5f);
    adjust_legacy_depth(amplified, 2.f);
    adjust_legacy_depth(unchanged, 1.f);
    check(reduced.front() > depth.front() && reduced.back() < depth.back());
    check(amplified.front() < depth.front() && amplified.back() > depth.back() &&
          amplified.front() >= 0.f && amplified.back() <= 1.f);
    check(unchanged == depth);
    const std::array device_depth{.960f, .970f, .980f, .990f, 1.f};
    Options display;
    display.legacy_depth_strength = 10.f;
    display.legacy_depth_fixed_range = true;
    auto full_range = device_depth;
    check(prepare_legacy_depth(full_range, device_depth.front(), device_depth.back(), display, true));
    check(full_range.front() == 0.f && full_range.back() > .99999f &&
          full_range[1] > 0.f && full_range[3] < 1.f);
    display.legacy_depth_strength = 1.f;
    display.legacy_depth_near = .25f;
    display.legacy_depth_far = .75f;
    auto windowed = device_depth;
    check(prepare_legacy_depth(windowed, device_depth.front(), device_depth.back(), display, true));
    check(windowed[1] < .00001f && std::abs(windowed[2] - .5f) < .00001f && windowed[3] > .99999f);
    auto windowed_guide = device_depth;
    check(prepare_legacy_depth(windowed_guide, device_depth.front(), device_depth.back(), display, false) &&
          std::abs(windowed_guide[1] - device_depth.front()) < .00001f &&
          std::abs(windowed_guide[3] - device_depth.back()) < .00001f);
    display.legacy_depth_near = 0.f;
    display.legacy_depth_far = 1.f;
    auto full_guide = device_depth;
    check(prepare_legacy_depth(full_guide, device_depth.front(), device_depth.back(), display, false) &&
          full_guide == device_depth);
    display.legacy_depth_fixed_range = false;
    auto neutral_guide = device_depth;
    check(prepare_legacy_depth(neutral_guide, device_depth.front(), device_depth.back(), display, false) &&
          neutral_guide == device_depth);
    display.legacy_depth_strength = 10.f;
    auto tuned_guide = device_depth;
    check(prepare_legacy_depth(tuned_guide, device_depth.front(), device_depth.back(), display, false) &&
          tuned_guide.front() == device_depth.front() && tuned_guide.back() == device_depth.back() &&
          tuned_guide[1] < device_depth[1] && tuned_guide[3] > device_depth[3]);
    check(!prepare_legacy_depth(tuned_guide, 1.f, 1.f, display, true));
    DepthFreshness freshness;
    constexpr unsigned test_width = 24, test_height = 14;
    std::array<float, test_width * test_height> test_depth{};
    std::array<uint8_t, test_width * test_height * 4> first{}, lighter{}, moved{};
    for (unsigned y = 0; y < test_height; ++y) {
        for (unsigned x = 0; x < test_width; ++x) {
            const unsigned pixel = y * test_width + x;
            const uint8_t gray = (x + y) % 2 ? 140 : 32;
            test_depth[pixel] = .9f + float(x) * .001f;
            for (unsigned channel = 0; channel < 3; ++channel) {
                first[pixel * 4 + channel] = gray;
                lighter[pixel * 4 + channel] = gray + 24;
                moved[pixel * 4 + channel] = 172 - gray;
            }
        }
    }
    check(freshness.accept(first, test_depth, test_width, test_height));
    check(!freshness.accept(first, test_width, test_height, {}, 0, test_height));
    check(freshness.accept(lighter, test_depth, test_width, test_height));
    check(!freshness.accept(moved, test_depth, test_width, test_height));
    test_depth[0] += .01f;
    check(freshness.accept(moved, test_depth, test_width, test_height));
    freshness.reset();
    check(freshness.accept(first, test_depth, test_width, test_height));
    std::array<uint8_t,12> preview{};
    preview_legacy_depth(depth, depth.front(), depth.back(), preview);
    check(preview[0] == 255 && preview[4] > 0 && preview[4] < 255 && preview[8] == 0 && preview[11] == 255);
    auto remapped = depth;
    remap_legacy_depth(remapped, .90f, .96f);
    check(remapped[0] == 0.f && remapped[1] > .66f && remapped[1] < .67f && remapped[2] == 1.f);
    preview_legacy_depth(remapped, 0.f, 1.f, preview, true);
    check(preview[0] == 0 && preview[4] >= 169 && preview[4] <= 171 && preview[8] == 255);
    preview_legacy_depth({}, 0.f, 1.f, preview, true);
    check(preview[0] == 0 && preview[4] == 0 && preview[8] == 0 && preview[11] == 255);
    const auto before=helper_motion_json(p);
    for(auto invalid:std::vector<nlohmann::json>{
        {{"MotionEstimation",1}},{{"MotionProvider",8}},{{"MotionProvider",1.5}},
        {{"MotionProvider",uint64_t(1)<<32}},
        {{"FlowAnalysisHeight",63}},{{"FlowAnalysisHeight",2161}},
        {{"FlowConsistencyPixels",0}},{{"FlowCostThreshold",1001}},
        {{"MotionDiagnosticView",3}},{{"SceneDetection","true"}},{{"LegacyDepth",5}},{{"LegacyDepth",1.5}},
        {{"LegacyDepthStrength",10.001}},{{"LegacyDepthPreview",1}},{{"LegacyDepthInvert",1}},
        {{"LegacyDepthNear",.97}},{{"LegacyDepthFar",.9}},{{"LegacyDepthFixedRange",1}}}){
        const auto original=invalid;bool refused=false;
        try{split_helper_motion_patch(p,invalid);}catch(const std::exception&){refused=true;}
        check(refused&&helper_motion_json(p)==before&&invalid==original);
    }
    FeedGuideState pending{1,1920.f,-1080.f};check(feed_guides_valid(pending));
    check(feed_take_guides(pending).engine_motion==1);check(!feed_take_guides(pending).engine_motion);
    check(!feed_guides_valid({1,std::numeric_limits<float>::quiet_NaN(),1}));
    check(!feed_guides_valid({2,1,1}));
    check(feed_protocol_compatible(10)&&feed_protocol_compatible(11)&&feed_protocol_compatible(12)&&
          feed_protocol_compatible(13)&&feed_protocol_compatible(14)&&feed_protocol_compatible(15)&&
          feed_protocol_compatible(16) && !feed_protocol_compatible(17));
    {
        // Fused depth stats must match the validation/range they replaced.
        const std::array sample{.25f, .5f, 1.f, 0.f};
        const auto stats = depth_range_stats(sample);
        check(stats.valid && stats.minimum == 0.f && stats.maximum == 1.f);
        check(!depth_range_stats({}).valid);
        check(!depth_range_stats(std::array{std::numeric_limits<float>::quiet_NaN()}).valid);
        check(!depth_range_stats(std::array{1.5f}).valid);
        check(!depth_range_stats(std::array{-0.5f}).valid);
    }
    phoenix_addon::phoenix::Settings settings;
    apply_live_tuning_patch(settings,{{"ForceReset",true},{"ResetAfterSwitch",3},{"ResetEvery",7}});
    check(settings.force_reset&&settings.reset_after_switch==3&&settings.reset_every==7);
    check(live_tuning_json(settings).at("ForceReset")==true);
    std::cout<<nlohmann::json{{"passed",true},{"checks",checks},{"bits",sizeof(void*)*8}}.dump()<<'\n';
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
