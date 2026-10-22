#pragma once
#include "phoenix_native.h"
#include "../third_party/nlohmann/json.hpp"
#include "../src/native/model/weight_source.h"
#include <DirectXPackedVector.h>
#include <fstream>
#include <filesystem>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace phoenix_cpp {
// Owned by an asynchronous capture worker. No driver loading, host copies or
// inference from this class belongs on the game's command-recording thread.
class PipelineClient {
    using Json=nlohmann::json;
    HMODULE library_=nullptr;PhoenixNativeSession* session_=nullptr;
    decltype(&phoenix_native_create) create_=nullptr;
    decltype(&phoenix_native_destroy) destroy_=nullptr;
    decltype(&phoenix_native_reconfigure) reconfigure_=nullptr;
    decltype(&phoenix_native_set_tuning) tuning_=nullptr;
    decltype(&phoenix_native_frame_host) frame_=nullptr;
    decltype(&phoenix_native_set_frame_info) identity_=nullptr;
    decltype(&phoenix_native_set_latency) latency_=nullptr;
    decltype(&phoenix_native_status) status_=nullptr;
    decltype(&phoenix_native_error) error_=nullptr;
    Json config_=nullptr;
    std::string runtime_digest_;
    std::vector<uint16_t> input_,output_;
    template<class T>void load(T& target,const char* name){target=reinterpret_cast<T>(GetProcAddress(library_,name));if(!target)throw std::runtime_error(name);}
    static std::string digest(const std::filesystem::path& path){
        std::ifstream file(path,std::ios::binary|std::ios::ate);if(!file)throw std::runtime_error("Portable pipeline dependency is missing");
        const auto bytes=file.tellg();if(bytes<=0||uint64_t(bytes)>128ull*1024*1024)throw std::runtime_error("Invalid portable dependency size");
        std::vector<uint8_t> data(size_t(bytes),0);file.seekg(0);file.read(reinterpret_cast<char*>(data.data()),bytes);
        if(!file)throw std::runtime_error("Cannot read portable pipeline dependency");
        return phoenix_model::weight_digest(data.data(),data.size());
    }
public:
    static bool supported(const std::filesystem::path& root){
        if(sizeof(void*)!=8)return false;
        HMODULE library=LoadLibraryExW((root/L"phoenix_native.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
        if(!library)return false;
        auto contract=reinterpret_cast<unsigned(*)()>(GetProcAddress(library,"phoenix_native_portable_contract"));
        const bool result=contract&&contract()>=1;FreeLibrary(library);return result;
    }
    PipelineClient(const std::filesystem::path& root){
        library_=LoadLibraryExW((root/L"phoenix_native.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
        if(!library_)throw std::runtime_error("Install the shared native pipeline SDK to use these portable stages");
        try{
            runtime_digest_=digest(root/L"phoenix_runtime_cpp.dll");
            auto contract=reinterpret_cast<unsigned(*)()>(GetProcAddress(library_,"phoenix_native_portable_contract"));
            if(!contract||contract()<1)throw std::runtime_error("Update the native pipeline SDK before using portable frame services");
            load(create_,"phoenix_native_create");load(destroy_,"phoenix_native_destroy");load(frame_,"phoenix_native_frame_host");
            load(reconfigure_,"phoenix_native_reconfigure");load(tuning_,"phoenix_native_set_tuning");
            load(identity_,"phoenix_native_set_frame_info");load(latency_,"phoenix_native_set_latency");
            load(status_,"phoenix_native_status");load(error_,"phoenix_native_error");
        }catch(...){FreeLibrary(library_);library_=nullptr;throw;}
    }
    ~PipelineClient(){if(session_&&destroy_(session_)!=0)library_=nullptr;if(library_)FreeLibrary(library_);}
    PipelineClient(const PipelineClient&)=delete;
    Json status(){
        if(!session_)return {{"portable_pipeline_contract",2},{"pipeline_frame_services",true}};
        const auto size=status_(session_,nullptr,0);if(!size||size>4*1024*1024)throw std::runtime_error("Invalid portable pipeline status");
        std::string text(size,'\0');if(status_(session_,text.data(),size)!=size)throw std::runtime_error(error_());
        auto result=Json::parse(text.c_str());result["portable_pipeline_contract"]=2;result["pipeline_frame_services"]=true;
        result["device"]=config_.at("portable_runtime").at("device");
        result["capabilities"]["granular_controls"]=true;
        result["pipeline_age_scope"]="capture worker handoff through shared pipeline completion; excludes capture/readback before handoff and display scanout";
        result["pipeline_telemetry"]["capture_to_output_ms"]=result.at("source_to_sdk_completion_ms");
        result["pipeline_telemetry"]["motion_provider"]=result.at("motion_provider");
        result["pipeline_telemetry"]["sampled_gpu_stages"]=false;
        result["network_schedule"]={{"transformer_blocks",result.at("tuning").at("central_blocks")},
            {"estimate_ms",config_.value("neural_budget_clock",std::string{})=="backend_gpu"?result.at("gpu_processing_ms"):result.at("complete_call_ms")}};
        return result;
    }
    std::vector<float> frame(const std::filesystem::path& root,const Json& runner,const std::vector<float>& color,
        unsigned width,unsigned height,unsigned ow,unsigned oh,const Json& metadata,
        uint64_t sequence,int64_t timestamp_ns,double source_age_ms,const std::string& tiny_model={}){
        if(color.size()!=size_t(width)*height*3)throw std::runtime_error("Portable pipeline input size differs");
        auto portable=runner;portable["runtime_root"]=phoenix_model::source_path_string(root);
        portable["runtime_sha256"]=runtime_digest_;
        const auto controls=metadata.value("controls",Json::object());auto neural=controls;
        neural["network_passes"]=metadata.value("passes",1u);
        neural["central_blocks"]=metadata.value("transformer_blocks",8u);
        neural["repeats"]=metadata.value("repeats",std::array<int,8>{1,1,1,1,1,1,1,1});
        const auto budget=metadata.value("adaptive_budget",Json::object());
        if(budget.value("enabled",false)){
            neural["minimum_central_blocks"]=budget.value("minimum_blocks",1u);
            if(budget.value("mode",std::string("gpu_time"))=="gpu_time")neural["max_processing_ms"]=budget.at("target_ms");
            else neural["target_fps"]=budget.at("target_fps");
        }
        Json config{{"width",width},{"height",height},{"output_width",ow},{"output_height",oh},
            {"weight_source",runner.at("weight_source")},{"portable_runtime",portable},
            {"engine_motion_valid",false},{"pipeline",metadata.at("pipeline")},{"tuning",neural}};
        if(budget.value("enabled",false)&&budget.value("mode",std::string("gpu_time"))=="gpu_time"){
            if(runner.at("backend")=="cpu")throw std::runtime_error("GPU-time budgets require a GPU; use the pipeline age controller on CPU");
            config["neural_budget_clock"]="backend_gpu";
        }
        if(!tiny_model.empty())config["tiny_model_path"]=tiny_model;
        if(config!=config_){
            auto before=config_.is_object()?config_:Json::object();auto after=config;before.erase("tuning");after.erase("tuning");
            if(session_&&before==after){if(tuning_(session_,neural.dump().c_str()))throw std::runtime_error(error_());}
            else if(session_){if(reconfigure_(&session_,config.dump().c_str()))throw std::runtime_error(error_());}
            else {session_=create_(config.dump().c_str(),nullptr,nullptr);if(!session_)throw std::runtime_error(error_());}
            config_=std::move(config);
        }
        input_.resize(size_t(width)*height*4);output_.resize(size_t(ow)*oh*4);
        for(size_t p=0;p<color.size()/3;++p){for(unsigned c=0;c<3;++c){
            const float value=color[p*3+c];if(!std::isfinite(value)||value<0||value>1)throw std::runtime_error("Invalid portable proxy input");
            input_[p*4+c]=DirectX::PackedVector::XMConvertFloatToHalf(value);
        }input_[p*4+3]=0x3c00;}
        PhoenixNativeFrameInfo info{sizeof(info),1,0,sequence,timestamp_ns,0,0};
        if(identity_(session_,&info)||latency_(session_,std::clamp(source_age_ms,0.,60000.),0))throw std::runtime_error(error_());
        PhoenixNativeControls options{sizeof(options),unsigned(metadata.value("reset",false)),1,1,1,1,-1,0,1};
        if(frame_(session_,input_.data(),input_.size()*2,size_t(width)*8,nullptr,0,0,
            output_.data(),output_.size()*2,size_t(ow)*8,&options))throw std::runtime_error(error_());
        std::vector<float> output(size_t(ow)*oh*3);
        for(size_t p=0;p<output.size()/3;++p)for(unsigned c=0;c<3;++c)output[p*3+c]=DirectX::PackedVector::XMConvertHalfToFloat(output_[p*4+c]);
        return output;
    }
};
}
