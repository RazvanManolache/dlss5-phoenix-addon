#include "../../native_runtime/native_execution_client.h"
#include <iostream>
#include <fstream>
using namespace phoenix_cpp;
using Json=nlohmann::json;
int main(int argc,char** argv) {
    try {
        if(argc!=4)throw std::runtime_error("RUNTIME_ROOT CONFIG_JSON REPORT_JSON");
        Json config;std::ifstream(argv[2])>>config;
        if(config.contains("config"))config=config.at("config").get<Json>();
        const auto root=std::filesystem::absolute(std::filesystem::u8path(argv[1]));
        constexpr unsigned w=64,h=48;const size_t pixels=size_t(w)*h;
        std::vector<float> color(pixels*3),motion(pixels*2,0),depth(pixels,.5f),valid(pixels,1),
            mask(pixels*4,1),ui(pixels*4,0),alpha(pixels,.25f),bb(pixels*3,.3f),field(pixels*4,0),output(pixels*3);
        for(size_t p=0;p<pixels;++p) {
            mask[p*4]=p%w<w/2?0.f:1.f;
            ui[p*4]=ui[p*4+1]=ui[p*4+2]=.1f;ui[p*4+3]=.25f;
        }
        Json report={{"passed",false},{"bitness",sizeof(void*)*8},{"gpu_used",true},{"routes",Json::array()}};
        std::vector<std::vector<float>> across;
        std::vector<std::string> modes=sizeof(void*)==8?std::vector<std::string>{"in_process","broker"}:std::vector<std::string>{"broker"};
        for(const auto& mode:modes) {
            config["execution"]=mode;config["broker_protocol"]=2;
            std::vector<std::vector<float>> reference;
            for(unsigned trial=0;trial<2;++trial) {
                NativeSession session;session.open(root,config,w,h);
                for(unsigned f=0;f<2;++f) {
                    for(size_t i=0;i<color.size();++i)color[i]=.1f+float((i*17+f*13)%700)/1000.f;
                    Json metadata={{"reset",f==0},{"has_motion",true},{"has_depth",true},
                        {"has_history_validity",true},{"controls",{{"sampling_mode","advancing"}}}};
                    if(trial&&f==1) {
                        mask[1]=NAN;auto rejected=metadata;rejected["reset"]=true;
                        if(session.frame_advanced(color.data(),motion.data(),depth.data(),valid.data(),rejected.dump(),output.data(),
                            mask.data(),ui.data(),alpha.data(),bb.data(),field.data())!=-1)throw std::runtime_error("Rejected frame accepted");
                        mask[1]=1;
                    }
                    if(session.frame_advanced(color.data(),motion.data(),depth.data(),valid.data(),metadata.dump(),output.data(),
                        mask.data(),ui.data(),alpha.data(),bb.data(),field.data()))throw std::runtime_error(session.error());
                    if(!std::all_of(output.begin(),output.end(),[](float v){return std::isfinite(v);}))throw std::runtime_error("Nonfinite transport result");
                    if(!trial)reference.push_back(output);
                    else if(reference[f]!=output)throw std::runtime_error("Rejected request changed continuation");
                    if(!trial&&mode=="in_process")across.push_back(output);
                    else if(!trial&&!across.empty()&&across[f]!=output)throw std::runtime_error("Local/broker output mismatch");
                }
            }
            report["routes"].push_back({{"execution",mode},{"protocol",2},{"all_optional_planes",true},
                {"rejected_request_preserves_continuation",true},
                {"last_output_sha256",phoenix_model::weight_digest(reinterpret_cast<const uint8_t*>(reference.back().data()),reference.back().size()*sizeof(float))}});
        }
        // A new helper must still consume the fixed version-1 layout.
        config["execution"]="broker";config["broker_protocol"]=1;
        { NativeSession session;session.open(root,config,w,h);
          if(session.frame(color.data(),nullptr,nullptr,nullptr,"{\"reset\":true}",output.data()))throw std::runtime_error(session.error()); }
        if(std::filesystem::exists(root/"phoenix_native_cpp.json")) {
            config["execution"]="broker";config["broker_protocol"]=2;
            const Json metadata={{"reset",true},{"has_motion",true},{"has_depth",true},
                {"has_history_validity",true},{"controls",{{"sampling_mode","advancing"}}}};
            std::vector<float> expected(pixels*3);
            {NativeSession session;session.open(root,config,w,h);
                if(session.frame_advanced(color.data(),motion.data(),depth.data(),valid.data(),metadata.dump(),expected.data(),
                    mask.data(),ui.data(),alpha.data(),bb.data(),field.data()))throw std::runtime_error(session.error());}
            using namespace phoenix_addon::phoenix::portable;
            configure_native_runtime(root);
            const auto device=config.at("device").get<std::string>();
            WorkerClient worker(w,h,{},{},{},std::wstring(device.begin(),device.end()),{}, {},L"auto",{},{},
                phoenix_model::WeightSource::config(config.at("weight_source")));
            const auto deadline=GetTickCount64()+15000;
            while(!worker.status().value("initialized",false)&&GetTickCount64()<deadline)Sleep(1);
            if(!worker.submit(1,9,color.data(),motion.data(),depth.data(),metadata,valid.data(),mask.data(),
                ui.data(),alpha.data(),bb.data(),field.data()))throw std::runtime_error("Worker submission failed: "+worker.status().dump());
            for(auto* plane:{&color,&motion,&depth,&valid,&mask,&ui,&alpha,&bb,&field})
                std::fill(plane->begin(),plane->end(),NAN);
            uint64_t serial=0;uint32_t generation=0;Json answer;
            while(!worker.poll(serial,generation,output,answer)&&GetTickCount64()<deadline)Sleep(1);
            if(serial!=1||generation!=9||!answer.value("ok",false)||output!=expected)
                throw std::runtime_error("Worker owned snapshot mismatch: "+answer.dump());
            report["worker_mutated_caller_planes_after_submit"]=true;
            report["worker_snapshot_output_exact"]=true;
        }
        report["legacy_protocol_1"]=true;report["passed"]=true;
        std::ofstream(argv[3])<<report.dump(2)<<'\n';std::cout<<report.dump(2)<<'\n';
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
