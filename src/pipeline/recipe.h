#pragma once
#include "json.h"
#include <vector>

namespace phoenix_pipeline {
// Shared native controls. Repeated blocks live inside one network evaluation;
// whole-network passes and fusion branches each own a temporal stage instance.
struct Tuning {
    float intensity=1, tone=1, structure=1, skin=-1, color_strength=1;
    unsigned style=0, passes=1;
    unsigned blocks=8, minimum_blocks=1;
    float target_fps=0,max_time_ms=0;
    bool auto_mask=false, temporal=true;
    float neural_scale=100;
    std::array<int,8> repeats{1,1,1,1,1,1,1,1};
    bool operator==(const Tuning&)const=default;
    static Tuning read(const nlohmann::json& j){
        if(!j.is_object())throw std::invalid_argument("Neural tuning must be an object");
        Tuning t;
        t.intensity=j.value("intensity",1.f);t.tone=j.value("local_tone",1.f);
        t.structure=j.value("local_structure",1.f);t.skin=j.value("skin_structure",-1.f);
        t.color_strength=j.value("color_strength",1.f);t.style=j.value("style",0u);
        t.passes=j.value("network_passes",1u);t.auto_mask=j.value("auto_mask",false);
        t.blocks=j.value("central_blocks",8u);t.minimum_blocks=j.value("minimum_central_blocks",1u);
        t.target_fps=j.value("target_fps",0.f);t.max_time_ms=j.value("max_processing_ms",0.f);
        t.temporal=j.value("temporal",true);t.neural_scale=j.value("neural_scale",100.f);
        t.repeats=j.value("repeats",t.repeats);
        auto bounded=[](float v,float a,float b){return std::isfinite(v)&&v>=a&&v<=b;};
        if(!bounded(t.intensity,0,2)||!bounded(t.tone,0,2)||!bounded(t.structure,0,2)||
           !bounded(t.skin,-1,2)||!bounded(t.color_strength,0,1)||t.style>2||t.passes<1||t.passes>10||
           !bounded(t.neural_scale,10,100)||t.blocks<1||t.blocks>8||t.minimum_blocks<1||t.minimum_blocks>t.blocks||
           !bounded(t.target_fps,0,1000)||!bounded(t.max_time_ms,0,1000))throw std::invalid_argument("Invalid neural tuning");
        for(int repeat:t.repeats)if(repeat<0||repeat>4)throw std::invalid_argument("Central block repetitions must be between 0 and 4");
        return t;
    }
    nlohmann::json json()const{return {{"intensity",intensity},{"local_tone",tone},{"local_structure",structure},
        {"skin_structure",skin},{"color_strength",color_strength},{"style",style},{"network_passes",passes},
        {"auto_mask",auto_mask},{"temporal",temporal},{"neural_scale",neural_scale},{"repeats",repeats},
        {"central_blocks",blocks},{"minimum_central_blocks",minimum_blocks},{"target_fps",target_fps},{"max_processing_ms",max_time_ms}};}
};
struct ChainStep {Stage stage;unsigned width,height;};
inline std::vector<ChainStep> plan(const Options& p,unsigned width,unsigned height,unsigned output_width,unsigned output_height){
    if(!p.valid()||!width||!height||!output_width||!output_height)throw std::invalid_argument("Invalid processing-chain geometry");
    std::vector<ChainStep> result;
    if(!p.chain_enabled){result.push_back({Stage::Neural,width,height});result.push_back({Stage::Scale,output_width,output_height});return result;}
    if(!p.enabled[unsigned(Stage::Scale)]&&(width!=output_width||height!=output_height))
        throw std::invalid_argument("Enable the scaling stage when source and output dimensions differ");
    for(int selected:p.order){
        const auto stage=Stage(selected);
        if(!p.enabled[unsigned(stage)]||(stage==Stage::Fusion&&!p.fusion))continue;
        if(stage==Stage::Scale){width=output_width;height=output_height;}
        result.push_back({stage,width,height});
    }
    return result;
}
// Keep the original executor for an equivalent neural -> reconstruction path.
// Merely opening the chain UI must not add copies, queue waits or model owners.
inline bool default_chain(const Options& p,unsigned width,unsigned height,unsigned ow,unsigned oh){
    if(!p.chain_enabled)return true;
    if(!p.enabled[unsigned(Stage::Neural)]||(p.fusion&&p.enabled[unsigned(Stage::Fusion)]))return false;
    if(p.enabled[unsigned(Stage::Color)]&&(p.color_brightness!=0||p.color_contrast!=1||p.color_saturation!=1||p.color_gamma!=1))return false;
    if(!p.enabled[unsigned(Stage::Scale)]&&(width!=ow||height!=oh))return false;
    bool neural=false;
    for(const auto step:plan(p,width,height,ow,oh)){
        if(step.stage==Stage::Neural)neural=true;
        if(step.stage==Stage::Scale&&!neural&&(width!=ow||height!=oh||p.scaler!=0))return false;
    }
    return true;
}
inline bool default_tuning(Tuning tuning){
    tuning.target_fps=0;tuning.max_time_ms=0;tuning.minimum_blocks=1;
    return tuning==Tuning{};
}
// The original executor already supports block count and block repetitions.
// Those controls alone do not require extra stage textures or queue waits.
inline bool direct_tuning(Tuning tuning){
    tuning.blocks=8;tuning.repeats.fill(1);
    return default_tuning(tuning);
}
// Reuse a preceding full-size neural prediction only while intervening stages
// preserve its pixels. Lower-resolution branches consume that prediction's
// original input; otherwise fusion evaluates its own full-resolution branch.
inline bool fusion_reuses_neural(const Options& p,unsigned width,unsigned height,unsigned ow,unsigned oh){
    bool ready=false;
    for(const auto step:plan(p,width,height,ow,oh)){
        if(step.stage==Stage::Neural)ready=true;
        else if(step.stage==Stage::Fusion)return ready;
        else if(step.stage==Stage::Color&&(p.color_brightness!=0||p.color_contrast!=1||p.color_saturation!=1||p.color_gamma!=1))ready=false;
        else if(step.stage==Stage::Scale){
            if(step.width!=width||step.height!=height||p.scaler!=0)ready=false;
            width=step.width;height=step.height;
        }
    }return false;
}
} // namespace phoenix_pipeline
