// Opt-in isolated GPU profiler. It never launches or modifies a game.
#define NOMINMAX
#include "../../src/integration/compatibility/wow64/graph32.h"
#include "wow64_graphics_context.h"
#include <atomic>
#include <iostream>

static std::string checksum(const std::vector<uint8_t>& pixels) {
    BCRYPT_HASH_HANDLE hash=nullptr;unsigned char digest[32]{};char text[65]{};
    phoenix_bundle::require(BCryptCreateHash(BCRYPT_SHA256_ALG_HANDLE,&hash,nullptr,0,nullptr,0,0)>=0,"Profile hash create");
    const auto added=BCryptHashData(hash,const_cast<PUCHAR>(pixels.data()),ULONG(pixels.size()),0);
    const auto finished=added>=0?BCryptFinishHash(hash,digest,sizeof(digest),0):added;
    BCryptDestroyHash(hash);phoenix_bundle::require(finished>=0,"Profile hash failed");
    for(unsigned i=0;i<32;++i)sprintf_s(text+i*2,3,"%02x",unsigned(digest[i]));
    return text;
}

int profile(int argc,wchar_t** argv) try {
    using Json=nlohmann::json;
    phoenix_bundle::require(argc==3||argc==4,"runtime, external model source and optional diagnostic matrix flag required");
    auto source=phoenix_model::WeightSource::config(phoenix_bundle::json_file(argv[2]));
    phoenix_compat::Settings settings;settings.mode=phoenix_addon::phoenix::ours;
    settings.model_id=source.model().id;
    const bool diagnosis=argc==4;
    const bool sampled=diagnosis&&std::wstring_view(argv[3])==L"--diagnose-sampled";
    if(diagnosis)settings.pipeline.sample_interval=sampled?30:1;
    constexpr unsigned width=1920,height=1080;
    auto root=std::filesystem::absolute(argv[1]);
    phoenix_wow64::FrameSession session(root,root/"native",source,width,height,settings,false);
    std::vector<uint8_t> input(size_t(width)*height*4,255),output;
    for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x) {
        auto p=(size_t(y)*width+x)*4;input[p]=uint8_t(x*7+y*3);
        input[p+1]=uint8_t((x/13+y/17)%2?190:50);input[p+2]=uint8_t(x+y*11);
    }
    Json runs=Json::array();uint64_t sequence=0;
    const std::vector<int> cases=diagnosis?std::vector<int>{0,1,2}:std::vector<int>{0,1};
    for(int selected:cases) {
        const int monitoring=diagnosis?2:selected*2;
        settings.pipeline.motion_enabled=!diagnosis||selected!=2;
        settings.pipeline.flow_height=diagnosis&&selected==1?90:180;
        settings.pipeline.monitoring=monitoring;Json samples=Json::array();
        const unsigned warmup=sampled?30:16,measured=sampled?240:32;
        for(unsigned frame=0;frame<warmup+measured;++frame) {
            phoenix_wow64::NativeFrameContext context{++sequence,int64_t(sequence)*16666667,frame==0};
            auto start=std::chrono::steady_clock::now();
            session.evaluate(input,settings,3,nullptr,&context,&output);
            auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            phoenix_bundle::require(output.size()==input.size(),"Incomplete profile output");
            if(frame>=warmup) {
                const auto& s=session.runtime_status();
                samples.push_back({{"wall_ms",elapsed},{"stages",s.value("frame_stages_ms",Json::object())},
                    {"transfer",s.value("transfer_stages_ms",Json::object())},
                    {"sdk",s.value("sdk_stages_ms",Json::object())},
                    {"motion_stages",s.value("motion_stages_ms",Json::object())},
                    {"worker",s.value("native_worker_stages_ms",Json::object())},
                    {"gpu_ms",s.value("gpu_processing_ms",0.)},
                    {"gpu_sampled",s.value("gpu_sampled_last_frame",false)},
                    {"batched_output",s.value("batched_output_submission",false)},
                    {"compact_status",s.value("compact_frame_status",false)},
                    {"motion_completion",s.value("motion_completion","legacy")},
                    {"motion_cpu_waits",s.value("motion_cpu_waits",Json())},
                    {"motion_valid",s.at("motion").at("valid")},{"half_buffer_bytes",session.host_half_buffer_bytes()}});
            }
        }
        runs.push_back({{"monitoring",monitoring},{"motion_enabled",settings.pipeline.motion_enabled},
            {"flow_height",settings.pipeline.flow_height},{"gpu_sample_interval",settings.pipeline.sample_interval},
            {"output_sha256",checksum(output)},{"samples",std::move(samples)}});
    }
    session.retire();
    std::cout<<Json{{"passed",true},{"model",source.model().id},{"width",width},{"height",height},
        {"inference_pid",GetCurrentProcessId()},{"helper_pid",0},{"runs",std::move(runs)}}.dump()<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<e.what();return 1;}

int wmain(int argc,wchar_t** argv) {
    phoenix_wow64_tests::OpenGlContext graphics;graphics.create();
    std::atomic<bool> done=false;int result=1;
    std::thread worker([&]{result=profile(argc,argv);done=true;});
    while(!done){graphics.render();MSG msg{};while(PeekMessageW(&msg,nullptr,0,0,PM_REMOVE))DispatchMessageW(&msg);Sleep(5);}
    worker.join();return result;
}
