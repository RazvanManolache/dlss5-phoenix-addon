#include "phoenix_native.h"
#include "../../src/pipeline/recipe.h"
#include <DirectXPackedVector.h>
#include <iostream>
#include <vector>
#include <array>
using Json=nlohmann::json;
using Image=std::vector<std::array<float,4>>;
namespace {
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(std::string(why)+": "+phoenix_native_error());}
float quantize(float value){return DirectX::PackedVector::XMConvertHalfToFloat(DirectX::PackedVector::XMConvertFloatToHalf(value));}
Image scale(const Image& source,unsigned sw,unsigned sh,unsigned w,unsigned h){
    auto sample=[&](float x,float y,unsigned c){
        const int ix=int(std::floor(x)),iy=int(std::floor(y));const float fx=x-std::floor(x),fy=y-std::floor(y);
        auto read=[&](int px,int py){return source[size_t(std::clamp(py,0,int(sh)-1))*sw+std::clamp(px,0,int(sw)-1)][c];};
        return std::lerp(std::lerp(read(ix,iy),read(ix+1,iy),fx),std::lerp(read(ix,iy+1),read(ix+1,iy+1),fx),fy);
    };
    Image result(size_t(w)*h);
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)for(unsigned c=0;c<4;++c){
        const float px=(float(x)+.5f)*sw/w-.5f,py=(float(y)+.5f)*sh/h-.5f;
        const float ox=std::max(float(sw)/w-1,0.f)*.25f,oy=std::max(float(sh)/h-1,0.f)*.25f;
        result[size_t(y)*w+x][c]=quantize((sample(px+ox,py+oy,c)+sample(px-ox,py-oy,c)+sample(px+ox,py-oy,c)+sample(px-ox,py+oy,c))*.25f);
    }return result;
}
Image color(Image values){for(auto& p:values){const float l=p[0]*.2126f+p[1]*.7152f+p[2]*.0722f;
    for(unsigned c=0;c<3;++c)p[c]=quantize(std::pow(std::clamp((std::lerp(l,p[c],.6f)-.5f)*1.7f+.65f,0.f,1.f),1/.8f));}return values;}
float error(const Image& a,const Image& b){check(a.size()==b.size(),"Image extent mismatch");float result=0;
    for(size_t p=0;p<a.size();++p)for(unsigned c=0;c<4;++c)result=std::max(result,std::abs(a[p][c]-b[p][c]));return result;}
struct Session{PhoenixNativeSession* value=nullptr;explicit Session(const Json& c):value(phoenix_native_create(c.dump().c_str(),nullptr,nullptr)){check(value,"Chain initialization");}
    ~Session(){if(value)phoenix_native_destroy(value);}};
Json status(PhoenixNativeSession* s){std::string text(phoenix_native_status(s,nullptr,0),'\0');phoenix_native_status(s,text.data(),text.size());return Json::parse(text.c_str());}
Image frame(PhoenixNativeSession* s,const Image& source,unsigned w,unsigned h,unsigned ow,unsigned oh,unsigned sequence=0){
    std::vector<uint16_t> input(source.size()*4),output(size_t(ow)*oh*4);
    for(size_t p=0;p<source.size();++p)for(unsigned c=0;c<4;++c)input[p*4+c]=DirectX::PackedVector::XMConvertFloatToHalf(source[p][c]);
    PhoenixNativeFrameInfo id{sizeof(id),1,1,sequence,int64_t(sequence)*16666667,0,0};check(phoenix_native_set_frame_info(s,&id)==0,"Describe chain frame");
    check(phoenix_native_frame_host(s,input.data(),input.size()*2,w*8,nullptr,0,0,output.data(),output.size()*2,ow*8,nullptr)==0,"Execute chain frame");
    Image result(size_t(ow)*oh);for(size_t p=0;p<result.size();++p)for(unsigned c=0;c<4;++c)result[p][c]=DirectX::PackedVector::XMConvertHalfToFloat(output[p*4+c]);return result;
}
}
int main(){try{
    constexpr unsigned w=41,h=23,ow=82,oh=46;
    Image input(w*h);for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){auto& p=input[size_t(y)*w+x];
        p={quantize(float((x*37+y*11)%256)/255),quantize(float((x*7+y*53)%256)/255),quantize(float((x*19+y*3)%256)/255),quantize(float((x+y)%7)/6)};}
    // Enable validation before the first WARP device: enabling the debug layer
    // after creating a device deliberately invalidates that existing device.
    Json neutral{{"width",w},{"height",h},{"spatial_only",true},{"warp",true},{"debug",true},{"pipeline",{{"ProcessingChain",true},{"PipelineMonitoring",0}}},
        {"tuning",phoenix_pipeline::Tuning{}.json()}};
    Session equivalent(neutral);check(status(equivalent.value).at("chain_execution")=="original executor (equivalent stages folded)","Neutral settings added unnecessary graph work");
    check(phoenix_native_set_tuning(equivalent.value,neutral["tuning"].dump().c_str())==0,"Identical tuning refused");
    check(status(equivalent.value).at("chain_execution")=="original executor (equivalent stages folded)","Identical tuning introduced graph work");
    neutral["tuning"]["central_blocks"]=4;neutral["tuning"]["repeats"]={1,0,2,1,1,1,1,1};
    Session granular(neutral);
    check(status(granular.value).at("chain_execution")=="original executor (equivalent stages folded)","Block-only tuning introduced graph work");
    frame(granular.value,input,w,h,w,h);
    check(phoenix_native_set_tuning(granular.value,neutral["tuning"].dump().c_str())==0,"Identical block tuning refused");
    const auto folded=status(granular.value);
    check(folded.at("frames")==1&&folded.at("completed_tuning").at("central_blocks")==4&&folded.at("chain_execution")=="original executor (equivalent stages folded)","Identical tuning discarded completed state");
    Json p{{"ProcessingChain",true},{"ChainNeural",false},{"ChainFusion",false},{"ChainStage1",2},{"ChainStage2",1},{"ChainStage3",0},{"ChainStage4",3},
        {"ChainBrightness",.15},{"ChainContrast",1.7},{"ChainSaturation",.6},{"ChainGamma",.8}};
    Json config{{"width",w},{"height",h},{"output_width",ow},{"output_height",oh},{"spatial_only",true},{"warp",true},{"debug",true},{"pipeline",p}};
    Session first(config);auto before=frame(first.value,input,w,h,ow,oh);
    auto expected_before=scale(color(input),w,h,ow,oh);auto alpha=scale(input,w,h,ow,oh);
    for(size_t i=0;i<alpha.size();++i)expected_before[i][3]=alpha[i][3];
    const auto error_before=error(before,expected_before);check(error_before<.002,"Color-before-scale CPU oracle");
    config["pipeline"]["ChainStage1"]=1;config["pipeline"]["ChainStage2"]=2;
    Session second(config);auto after=frame(second.value,input,w,h,ow,oh);
    auto expected_after=color(scale(input,w,h,ow,oh));
    const auto error_after=error(after,expected_after);check(error_after<.002,"Scale-before-color CPU oracle");
    check(error(before,after)>.01,"Reordering stages did not change the noncommuting result");
    Json bypass{{"ProcessingChain",true},{"ChainNeural",false},{"ChainScale",false},{"ChainColor",false},{"ChainFusion",false}};
    config["output_width"]=w;config["output_height"]=h;config["pipeline"]=bypass;
    Session off(config);check(error(frame(off.value,input,w,h,w,h),input)==0,"Disabled chain must be exact bypass");
    check(phoenix_native_set_pipeline(off.value,"{\"ChainStage1\":2,\"ChainStage2\":2}")!=0,"Invalid order was accepted");
    check(phoenix_native_set_tuning(off.value,"{\"network_passes\":0}")!=0,"Invalid network pass count accepted");
    check(phoenix_native_set_tuning(off.value,"{\"repeats\":[1,1,1,1,1,1,1,5]}")!=0,"Unsupported repeat count accepted");
    check(error(frame(off.value,input,w,h,w,h,1),input)==0,"Rejected settings damaged the working session");
    auto fusion=bypass;fusion["ChainFusion"]=true;fusion["MultiscaleFusion"]=true;fusion["FusionLayers"]=3;
    config["pipeline"]=fusion;config["tuning"]={{"network_passes",2}};
    Session fused(config);auto actual=frame(fused.value,input,w,h,w,h);
    auto half=scale(scale(input,w,h,21,12),21,12,w,h),three_quarters=scale(scale(input,w,h,31,17),31,17,w,h);
    Image expected=input;
    for(size_t i=0;i<input.size();++i){for(unsigned c=0;c<3;++c)expected[i][c]=quantize(std::lerp(quantize(std::lerp(input[i][c],half[i][c],.5f)),three_quarters[i][c],1.f/3));expected[i][3]=input[i][3];}
    check(error(actual,expected)<.002,"Multiscale fusion CPU oracle");
    const auto report=status(fused.value);check(report.at("independent_temporal_stages")==6,"Fusion passes share temporal instances");
    config["pipeline"]=bypass;config["pipeline"]["PipelineMonitoring"]=1;config.erase("tuning");
    Session aged(config);
    check(phoenix_native_set_latency(aged.value,42,7)==0,"Caller latency sample refused");
    check(phoenix_native_set_latency(aged.value,-1,0)!=0,"Negative caller age accepted");
    frame(aged.value,input,w,h,w,h);const auto latency=status(aged.value).at("pipeline_telemetry");
    check(latency.at("caller_latency_available")==true&&std::abs(latency.at("source_to_sdk_completion_ms").get<double>()-latency.at("processing_call_ms").get<double>()-42)<1e-6,"Decoder queue age lost");
    check(std::abs(latency.at("predicted_present_return_age_ms").get<double>()-latency.at("source_to_sdk_completion_ms").get<double>()-7)<1e-6,"Presentation tail not budgeted");
    frame(aged.value,input,w,h,w,h,1);check(!status(aged.value).at("caller_latency_available").get<bool>(),"Source age leaked into another frame");
    std::cout<<Json{{"passed",true},{"warp",true},{"stage_order_changes_pixels",true},{"disabled_exact_bypass",true},
        {"invalid_change_preserves_session",true},{"granular_tuning_folded",true},{"identical_tuning_preserves_executor",true},{"fusion_layers",3},{"independent_stages",6},{"color_scale_max_error",error_before},
        {"scale_color_max_error",error_after},{"fusion_max_error",error(actual,expected)}}.dump()<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
