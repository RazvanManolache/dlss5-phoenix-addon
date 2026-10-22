#include "../../src/integration/compatibility/host_client.h"
#include <iostream>
using namespace phoenix_compat;
void verify(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void retire(std::shared_ptr<HostState>& state){
    state->stop();const auto retired=std::weak_ptr<HostState>(state);state.reset();
    const auto end=GetTickCount64()+30000;while(!retired.expired()&&GetTickCount64()<end)Sleep(1);
    verify(retired.expired(),"Capture worker did not retire its model/SDK dependencies");
}
void wait_ready(const std::shared_ptr<HostState>& state,Phase phase){
    const auto end=GetTickCount64()+120000;
    while(state->phase()!=phase&&state->phase()!=Phase::Failed&&GetTickCount64()<end)Sleep(1);
    if(state->phase()!=phase)throw std::runtime_error(state->error());
}
int wmain(int argc,wchar_t** argv)try{
    verify(argc==3,"Pass a validated runtime root and external dual-model DLL");
    const auto root=fs::absolute(argv[1]),model=fs::absolute(argv[2]);
    verify(phoenix_cpp::PipelineClient::supported(root),"Shared portable pipeline capability missing");
    unsigned cases=0;
    for(const auto* model_id:{"original","small-fp8"}){
        Settings settings;settings.mode=phoenix_addon::phoenix::ours;settings.model_id=model_id;
        settings.pipeline.chain_enabled=true;settings.pipeline.enabled[0]=false;
        settings.pipeline.color_brightness=.1f;settings.pipeline.monitoring=1;
        phoenix_model::WeightSource source;source.dll=model;source.dll_sha256=phoenix_model::mapped_file_digest(model);source.model_id=model_id;
        FeedBuild build{};build.width=build.output_width=32;build.height=build.output_height=24;build.cpu_bridge=true;
        PortableOptions options{root,"cpu",true};
        auto state=std::make_shared<HostState>(settings);state->start(nullptr,source,settings,build,options);wait_ready(state,Phase::Ready);
        verify(state->ui_snapshot().at("runtime").at("pipeline_frame_services")==true,"Frontend did not receive the actual capability");
        std::vector<uint8_t> pixels(size_t(build.width)*build.height*4,80);
        verify(state->submit(1,true,pixels),"Cannot submit source frame");
        verify(!state->submit(2,true,pixels),"Capture worker admitted an unbounded second request");
        wait_ready(state,Phase::Finished);auto output=state->take();
        verify(output.size()==pixels.size(),"Wrong output extent");
        for(size_t p=0;p<output.size();p+=4){
            verify(std::abs(int(output[p])-106)<=1,"Color stage was not rendered by the shared pipeline");
            verify(output[p+3]==80,"Capture composition lost original opacity");
        }
        settings.processing.enabled=true;settings.processing.scale=50;settings.pipeline.scaler=1;settings.pipeline.color_brightness=0;
        retire(state);state=std::make_shared<HostState>(settings);state->start(nullptr,source,settings,build,options);wait_ready(state,Phase::Ready);
        verify(state->submit(2,true,pixels),"Cannot submit geometry-changing recipe");
        wait_ready(state,Phase::Finished);output=state->take();
        verify(output.size()==pixels.size(),"NIS changed the fixed capture output size");
        const auto info=state->ui_snapshot().at("runtime");
        verify(info.at("additional_scaler")=="NVIDIA Image Scaling","NIS selection was ignored");
        verify(info.at("width")==16&&info.at("output_width")==32,"Processing and output extents were conflated");
        for(size_t p=0;p<output.size();p+=4)verify(std::abs(int(output[p])-80)<=2&&output[p+3]==80,"Constant/opacity reconstruction differs");
        retire(state);
        // Compare the real capture worker against the independent public C API.
        // Canonicalize input through FP16, as the SDK's texture boundary does.
        settings.processing.enabled=false;settings.pipeline={};settings.pipeline.chain_enabled=true;
        settings.pipeline.enabled={true,false,false,false};settings.pipeline.monitoring=1;
        settings.experimental=true;settings.transformer_blocks=1;settings.cpu_threads=2;
        settings.controls.enabled=true;settings.controls.tone=.75f;settings.controls.structure=.65f;
        settings.controls.appearance=true;settings.controls.intensity=.7f;settings.controls.style=1;
        for(size_t p=0;p<pixels.size()/4;++p)for(unsigned c=0;c<3;++c)pixels[p*4+c]=uint8_t(30+(p*17+c*53)%180);
        auto rgb=portable_color(pixels);
        for(auto& value:rgb)value=DirectX::PackedVector::XMConvertHalfToFloat(DirectX::PackedVector::XMConvertFloatToHalf(value));
        const auto reference_config=Json{{"runtime_root",phoenix_model::source_path_string(root)},{"backend","cpu"},
            {"device","cpu"},{"cpu_threads",2},{"allow_unvalidated",true},{"weight_source",source.configuration()}};
        std::vector<float> expected(rgb.size());
        {phoenix_cpp::RuntimeApi oracle;oracle.open(root,reference_config.dump(),build.width,build.height);
            auto metadata=portable_metadata(settings,true,true);metadata.erase("pipeline");
            verify(oracle.frame(oracle.instance,rgb.data(),nullptr,nullptr,nullptr,metadata.dump().c_str(),expected.data())==0,"Independent neural reference failed");}
        const auto expected_pixels=portable_compose(pixels,expected,build.width,settings);
        state=std::make_shared<HostState>(settings);state->start(nullptr,source,settings,build,options);wait_ready(state,Phase::Ready);
        verify(state->submit(3,true,pixels),"Cannot submit actual neural frame");wait_ready(state,Phase::Finished);output=state->take();
        unsigned changed=0;
        for(size_t p=0;p<output.size();++p){verify(std::abs(int(output[p])-int(expected_pixels[p]))<=1,"Capture neural/appearance pixels differ from direct C API");changed+=output[p]!=pixels[p];}
        verify(changed>100,"Capture neural processing was bypassed");
        auto snapshot=state->ui_snapshot();const auto pid=snapshot.at("inference_pid").get<DWORD>();
        verify(snapshot.at("processing_host_pid")==GetCurrentProcessId()&&pid&&pid!=GetCurrentProcessId(),"Neural worker PID is misreported as its capture host");
        verify(snapshot.at("inference_pids").size()==1,"Single pass reports the wrong worker count");
        HANDLE worker=OpenProcess(SYNCHRONIZE,FALSE,pid);verify(worker&&WaitForSingleObject(worker,0)==WAIT_TIMEOUT,"Reported neural worker is not running");
        settings.controls.intensity=0;verify(state->update_tuning(settings),"Cannot update portable appearance");
        verify(state->submit(4,true,pixels),"Cannot submit zero-intensity frame");wait_ready(state,Phase::Finished);output=state->take();
        for(size_t p=0;p<output.size();++p)verify(std::abs(int(output[p])-int(pixels[p]))<=1,"Portable zero intensity did not restore the source");
        const auto last_pid=state->ui_snapshot().at("inference_pid").get<DWORD>();
        HANDLE last_worker=OpenProcess(SYNCHRONIZE,FALSE,last_pid);verify(last_worker,"Updated neural worker cannot be observed");
        retire(state);verify(WaitForSingleObject(worker,5000)==WAIT_OBJECT_0&&WaitForSingleObject(last_worker,5000)==WAIT_OBJECT_0,"Retired capture left its neural worker alive");
        CloseHandle(last_worker);CloseHandle(worker);
        ++cases;
    }
    std::cout<<"{\"passed\":true,\"models\":"<<cases<<",\"asynchronous_backbuffer\":true,\"bounded_queue\":true,\"color_oracle\":true,\"geometry_nis_constant_oracle\":true,\"neural_appearance_oracle\":true,\"zero_intensity_source_oracle\":true,\"worker_identity_and_retirement\":true,\"opacity_preserved\":true}\n";
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
