// One offline host-input -> completed host-output timing boundary for all engines.
#define main phoenix_existing_four_model_main
#include "four_model_probe.cpp"
#undef main
#include "../../native_runtime/phoenix_native.h"

struct UnifiedSdk {
    HMODULE module{};PhoenixNativeSession* session{};
    decltype(&phoenix_native_destroy) destroy{};
    decltype(&phoenix_native_frame_d3d12) frame{};
    decltype(&phoenix_native_error) error{};
    decltype(&phoenix_native_status) status{};
    UnifiedSdk(const Json& job,Gpu& gpu,unsigned w,unsigned h){
        const auto root=fs::u8path(job.at("runtime_root").get<std::string>());
        module=LoadLibraryExW((root/L"phoenix_native.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        require(module!=nullptr,"Unified SDK load failed");
        auto create=symbol<decltype(&phoenix_native_create)>(module,"phoenix_native_create");
        destroy=symbol<decltype(destroy)>(module,"phoenix_native_destroy");frame=symbol<decltype(frame)>(module,"phoenix_native_frame_d3d12");
        error=symbol<decltype(error)>(module,"phoenix_native_error");status=symbol<decltype(status)>(module,"phoenix_native_status");
        Json config={{"width",w},{"height",h},{"weight_source",job.at("weight_source")},{"bundle_path",job.at("bundle_path")},
            {"engine_motion_valid",true},{"pipeline",{{"MotionEstimation",false},{"SceneDetection",false},{"PipelineMonitoring",0},{"ProcessingChain",false}}},
            {"tuning",{{"auto_mask",true},{"local_tone",1.f},{"local_structure",1.f},{"skin_structure",-1.f},{"network_passes",1}}}};
        if(job.contains("portable_runtime"))config["portable_runtime"]=job.at("portable_runtime");
        session=create(config.dump().c_str(),gpu.device.Get(),gpu.queue.Get());require(session!=nullptr,error());
    }
    ~UnifiedSdk(){if(session&&destroy(session)!=0)module=nullptr;if(module)FreeLibrary(module);}
    void evaluate(Texture& color,Texture& motion,Texture& output){
        PhoenixNativeControls controls{sizeof(controls),1,1,1,1,1,-1,1,0};
        require(frame(session,color.resource.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            motion.resource.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,output.resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,&controls)==0,error());
    }
    Json snapshot(){std::string data(status(session,nullptr,0),'\0');status(session,data.data(),data.size());return Json::parse(data.c_str());}
};
int main(int argc,char** argv){Json report{{"passed",false}};fs::path destination;
 try{
    require(argc==3,"JOB_JSON OUTPUT_JSON");const auto job=phoenix_bundle::json_file(argv[1]);destination=argv[2];
    const unsigned w=job.at("width"),h=job.at("height");const bool vendor=job.at("engine")=="nvidia";
    const auto input=read(fs::u8path(job.at("color_file").get<std::string>()));require(input.size()==size_t(w)*h*8,"Input dimensions differ");
    Gpu gpu(job.value("debug",false));auto color=texture(gpu.device.Get(),w,h,16,false),motion=texture(gpu.device.Get(),w,h,16,false,2),output=texture(gpu.device.Get(),w,h,16,true);
    auto upload=[&](Texture& image,const uint8_t* bytes,unsigned channels,D3D12_RESOURCE_STATES from){
        void* mapped=nullptr;D3D12_RANGE empty{};check(image.transfer->Map(0,&empty,&mapped));
        for(unsigned y=0;y<h;++y)memcpy(static_cast<uint8_t*>(mapped)+image.footprint.Offset+size_t(y)*image.footprint.Footprint.RowPitch,bytes+size_t(y)*w*channels*2,size_t(w)*channels*2);
        image.transfer->Unmap(0,nullptr);gpu.begin();if(from!=D3D12_RESOURCE_STATE_COPY_DEST)barrier(gpu.command.Get(),image.resource.Get(),from,D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION a{image.transfer.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,image.footprint},b{image.resource.Get(),D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0};
        gpu.command->CopyTextureRegion(&b,0,0,0,&a,nullptr);barrier(gpu.command.Get(),image.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);gpu.submit();
    };
    std::vector<uint8_t> zero(size_t(w)*h*4),result(input.size());upload(color,input.data(),4,D3D12_RESOURCE_STATE_COPY_DEST);upload(motion,zero.data(),2,D3D12_RESOURCE_STATE_COPY_DEST);
    std::unique_ptr<Nvidia> nvidia;std::unique_ptr<UnifiedSdk> sdk;
    const auto create_begin=Clock::now();
    if(vendor)nvidia=std::make_unique<Nvidia>(gpu,job,w,h,job.at("weight_source").at("model")=="original"?1:2);
    else sdk=std::make_unique<UnifiedSdk>(job,gpu,w,h);
    report["create_ms"]=ms(create_begin);
    auto frame=[&]{const auto start=Clock::now();
        upload(color,input.data(),4,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if(vendor){gpu.begin();nvidia->record(gpu,color,motion,output,true,w,h);gpu.submit();}else sdk->evaluate(color,motion,output);
        gpu.begin();barrier(gpu.command.Get(),output.resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION from{output.resource.Get(),D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0},to{output.transfer.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,output.footprint};
        gpu.command->CopyTextureRegion(&to,0,0,0,&from,nullptr);barrier(gpu.command.Get(),output.resource.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);gpu.submit();
        void* mapped=nullptr;D3D12_RANGE range{0,size_t(output.bytes)};check(output.transfer->Map(0,&range,&mapped));
        for(unsigned y=0;y<h;++y)memcpy(result.data()+size_t(y)*w*8,static_cast<uint8_t*>(mapped)+output.footprint.Offset+size_t(y)*output.footprint.Footprint.RowPitch,size_t(w)*8);
        D3D12_RANGE none{};output.transfer->Unmap(0,&none);const double elapsed=ms(start);
        for(size_t i=0;i<result.size();i+=2){uint16_t value;memcpy(&value,result.data()+i,2);require((value&0x7c00)!=0x7c00,"Nonfinite output");}
        return elapsed;};
    const unsigned warmup=job.value("warmup",3u),count=job.value("samples",15u);require(warmup>=1&&count>=3,"Insufficient samples");
    for(unsigned i=0;i<warmup;++i)frame();std::vector<double> samples;std::string first_hash;
    for(unsigned i=0;i<count;++i){samples.push_back(frame());const auto hash=phoenix_model::weight_digest(result.data(),result.size());if(i==0)first_hash=hash;require(hash==first_hash,"Reset output is not repeatable");}
    if(job.contains("output_file")){std::ofstream f(fs::u8path(job.at("output_file").get<std::string>()),std::ios::binary);f.write(reinterpret_cast<const char*>(result.data()),result.size());}
    report.update({{"passed",true},{"engine",job.at("engine")},{"model",job.at("weight_source").at("model")},{"width",w},{"height",h},
        {"input_sha256",phoenix_model::weight_digest(input.data(),input.size())},{"output_sha256",first_hash},{"reset_repeatable",true},
        {"warmup",warmup},{"samples_ms",samples},{"wall",stats(samples)},{"memory",gpu.memory()},
        {"scope","Host RGBA16F input -> upload -> one full reset neural pass -> readback -> completed host RGBA16F output; setup, allocation, validation and hashing excluded; zero motion, auto mask, neutral controls"}});
    if(sdk)report["status"]=sdk->snapshot();report["debug_errors"]=gpu.errors();require(report["debug_errors"]==0,"D3D12 debug error");
    std::ofstream(destination)<<report.dump(2)<<'\n';std::cout<<job.at("engine")<<" "<<job.at("weight_source").at("model")<<" "<<report["wall"]["median_ms"]<<" ms\n";return 0;
 }catch(const std::exception& error){report["error"]=error.what();if(!destination.empty())std::ofstream(destination)<<report.dump(2)<<'\n';std::cerr<<error.what()<<'\n';return 1;}}
