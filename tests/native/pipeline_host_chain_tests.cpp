#include "../../src/pipeline/host_chain.h"
#include <iostream>
#include <set>
using namespace phoenix_pipeline;
static void check(bool value,const char* text){if(!value)throw std::runtime_error(text);}
int main(){try{
    HostImage input(42,26,3);std::fill(input.pixels.begin(),input.pixels.end(),.2f);
    Options p;p.chain_enabled=true;p.fusion=true;p.enabled[unsigned(Stage::Fusion)]=true;
    std::set<std::string> stages;
    auto affine=[&](const std::string& key,const HostImage& source){
        stages.insert(key);auto result=source;
        for(auto& value:result.pixels)value=.4f+.5f*value;
        return result;
    };
    auto combined=host_chain(p,input,42,26,affine);
    for(float value:combined.pixels)check(std::abs(value-.5f)<1e-6f,"Fusion reprocessed a prediction");
    check(stages==std::set<std::string>{"neural","fusion1"},"Full prediction was not reused");
    stages.clear();p.enabled[unsigned(Stage::Neural)]=false;p.fusion_layers=3;
    auto standalone=host_chain(p,input,42,26,affine);
    for(float value:standalone.pixels)check(std::abs(value-.5f)<1e-6f,"Standalone fusion omitted full prediction");
    check(stages==std::set<std::string>{"fusion0","fusion1","fusion2"},"Fusion histories share a stage key");
    p.enabled[unsigned(Stage::Neural)]=true;p.fusion_layers=2;
    p.order={0,2,3,1};p.color_brightness=.1f;stages.clear();
    host_chain(p,input,42,26,affine);
    check(stages.contains("fusion0"),"Color treatment must invalidate prediction reuse");
    HostImage motion(42,26,2);for(size_t i=0;i<motion.pixels.size();i+=2){motion.pixels[i]=8;motion.pixels[i+1]=-4;}
    const auto reduced=host_resample(motion,21,13,HostSampling::Motion);
    check(reduced.pixels[0]==4&&reduced.pixels[1]==-2,"Motion units did not follow geometry");
    HostImage validity(42,26,1);std::fill(validity.pixels.begin(),validity.pixels.end(),1.f);validity.pixels[0]=0;
    const auto reduced_validity=host_resample(validity,7,5,HostSampling::Validity);
    check(reduced_validity.pixels[0]==0&&reduced_validity.pixels.back()==1,"Reduction hid rejected history");
    auto invalid=p;invalid.scaler=1;bool rejected=false;
    try{host_chain(invalid,input,42,26,affine);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"Unavailable scaler silently changed the recipe");
    std::cout<<nlohmann::json{{"passed",true},{"same_source_fusion",true},{"standalone_full_prediction",true},
        {"independent_stage_keys",true},{"motion_units",true},{"conservative_validity",true}}.dump()<<'\n';
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
