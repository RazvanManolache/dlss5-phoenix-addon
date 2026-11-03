// Isolated GPU test. No installed games, desktop input or external process.
#define NOMINMAX
#include "../../src/integration/compatibility/wow64/graph32.h"
#include "../../src/integration/compatibility/host_client.h"
#include <iostream>
#include <atomic>
#include "wow64_graphics_context.h"
#ifdef _WIN64
#error This test must be built for x86
#endif

int validate(int argc,wchar_t** argv) try {
    using namespace phoenix_wow64;
    using Json=nlohmann::json;
    phoenix_bundle::require(argc==3,"runtime directory and private weight-source JSON required");
    const auto root=std::filesystem::absolute(argv[1]);
    const auto source=phoenix_model::WeightSource::config(phoenix_bundle::json_file(argv[2]));
    const bool batched_package=phoenix_bundle::json_file(root/"bootstrap.json").value("library_batched_output",0)==1;
    const bool compact_package=phoenix_bundle::json_file(root/"bootstrap.json").value("library_compact_frame_status",0)==1;
    constexpr unsigned width=640,height=384;
    phoenix_compat::Settings settings;settings.mode=phoenix_addon::phoenix::ours;
    // The asynchronous worker selects from Settings, whereas FrameSession also
    // receives the explicit source. Keep both paths on the requested model.
    settings.model_id=source.model().id;
    FrameSession session(root,root/"native",source,width,height,settings,false);
    phoenix_bundle::require(session.frame_services(),"Runtime did not negotiate motion support");
    std::vector<uint8_t> pixels(size_t(width)*height*4,255);
    Json report={{"passed",false},{"caller_bits",32},{"inference_pid",GetCurrentProcessId()},
        {"helper_pid",0},{"model",source.model().id},{"retail_games_launched",false},{"frames",Json::array()}};
    unsigned valid_frames=0,history_frames=0;
    std::vector<uint8_t> packed_output;
    for(unsigned frame=0;frame<12;++frame) {
        // A known horizontal translation with texture at several scales.
        for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x) {
            const auto sx=(x+frame*2)%width;const auto p=(size_t(y)*width+x)*4;
            pixels[p]=uint8_t((sx*37+y*19+(sx/11+y/17)%2*83)&255);
            pixels[p+1]=uint8_t((sx/7+y/9)%2?210:43);
            pixels[p+2]=uint8_t((sx*3+y*7)&255);
        }
        if(frame==4)settings.pipeline.flow_height=90; // Live resolution edit, same worker.
        if(frame==6)settings.pipeline.scene_detection=true;
        if(frame==7)settings.force_reset=true;
        if(frame==8)settings.force_reset=false;
        if(frame==9)settings.pipeline.motion_source=1; // No engine guides in CPU transport.
        if(frame==10)settings.pipeline.motion_source=0;
        NativeFrameContext context{frame+1,int64_t(frame+1)*16666667,frame==0||frame==8};
        // Alternate the compact view and float comparison/refinement transport
        // within one session. Per-frame packet pointers/extents must reset.
        const bool packed_frame=frame%2==0;
        const auto output=session.evaluate(pixels,settings,3,nullptr,&context,packed_frame?&packed_output:nullptr);
        phoenix_bundle::require(packed_frame?output.empty()&&packed_output.size()==pixels.size():
            output.size()==size_t(width)*height*3,"Output extent mismatch");
        phoenix_bundle::require(std::all_of(output.begin(),output.end(),[](float v){return std::isfinite(v);}),"Nonfinite output");
        const auto& status=session.runtime_status();const auto& motion=status.at("motion");
        if(compact_package){
            phoenix_bundle::require(status.at("compact_frame_status").get<bool>(),"New package used full per-frame memory diagnostics");
            if(frame<6)phoenix_bundle::require(status.at("motion_completion")=="gpu_ordered"&&status.at("motion_cpu_waits")==0,
                "Native motion optimization introduced a CPU fence wait");
        }
        if(batched_package)phoenix_bundle::require(status.at("batched_output_submission")==packed_frame,
            "Compact frame did not use the verified combined submission, or float frame used it unexpectedly");
        phoenix_bundle::require(status.at("inference_pid")==GetCurrentProcessId(),"Inference left the x86 process");
        phoenix_bundle::require(status.at("pipeline").at("FlowAnalysisHeight")==settings.pipeline.flow_height,"Live motion resolution did not apply");
        if(motion.at("valid").get<bool>()) {
            ++valid_frames;phoenix_bundle::require(motion.at("source")==2,"Estimated vectors mislabeled");
        }
        if(motion.value("history_usable",false))++history_frames;
        if(frame==8)phoenix_bundle::require(motion.at("reset").get<bool>(),"Explicit reset lost");
        if(frame==7)phoenix_bundle::require(motion.at("reset").get<bool>(),"Force-reset control ignored");
        if(frame==9)phoenix_bundle::require(!motion.at("valid").get<bool>(),"Placeholder promoted to engine motion");
        report["frames"].push_back({{"sequence",frame+1},{"motion",motion}});
    }
    phoenix_bundle::require(valid_frames>=3&&history_frames>=2,"No sustained estimated motion/history");
    // Zero appearance intensity must reproduce the input in BOTH transports.
    // A policy-only test cannot catch a control silently dropped at the ABI.
    settings.controls.appearance=true; settings.controls.intensity=0; settings.controls.style=2;
    settings.force_reset=true;
    for (bool packed : {true,false}) {
        NativeFrameContext context{30,500000000,true};
        const auto output=session.evaluate(pixels,settings,3,nullptr,&context,packed?&packed_output:nullptr);
        if(packed) phoenix_bundle::require(packed_output==pixels,"Packed zero-intensity appearance did not preserve source");
        else for(size_t pixel=0;pixel<pixels.size()/4;++pixel)for(unsigned channel=0;channel<3;++channel)
            phoenix_bundle::require(std::abs(output[pixel*3+channel]-pixels[pixel*4+channel]/255.f)<.001f,
                "Half zero-intensity appearance did not preserve source");
        const auto& status=session.runtime_status();
        phoenix_bundle::require(status.at("effective_style")==2&&status.at("effective_intensity")==0,
            "Native appearance controls were not forwarded");
    }
    settings.controls.appearance=false; settings.force_reset=false;
    report["appearance_transports_verified"]=true;
    session.retire();
    // Exercise the exact asynchronous worker used by the add-on, including its
    // live motion patch and ResetHistory action, rather than only FrameSession.
    settings.pipeline.motion_source=0;settings.pipeline.flow_height=180;settings.pipeline.scene_detection=false;
    auto state=std::make_shared<phoenix_compat::HostState>(settings);
    FeedBuild build{};build.width=build.output_width=width;build.height=build.output_height=height;build.cpu_bridge=1;
    state->start(nullptr,source,settings,build,std::nullopt,phoenix_compat::NativeOptions{root,false});
    auto wait=[&](phoenix_compat::Phase desired) {
        const auto deadline=GetTickCount64()+120000;
        while(state->phase()!=desired) {
            if(state->phase()==phoenix_compat::Phase::Failed)throw std::runtime_error(state->error());
            phoenix_bundle::require(GetTickCount64()<deadline,"Motion worker timed out");Sleep(5);
        }
    };
    wait(phoenix_compat::Phase::Ready);Json worker=Json::array();
    double previous_wall_ms=0;
    unsigned worker_valid_frames=0,worker_history_frames=0;
    for(unsigned frame=1;frame<=12;++frame) {
        if(frame==4){settings.pipeline.flow_height=90;phoenix_bundle::require(state->update_tuning(settings),"Worker refused live motion resolution");}
        if(frame==9)phoenix_bundle::require(state->ui_action(phoenix_ui::Action::ResetHistory),"Worker refused history reset");
        phoenix_bundle::require(state->submit(frame,frame==1,pixels),"Motion worker refused frame");
        wait(phoenix_compat::Phase::Finished);
        const auto snapshot=state->ui_snapshot();
        phoenix_bundle::require(state->process_id()==GetCurrentProcessId(),"Motion worker used a helper process");
        phoenix_bundle::require(snapshot.at("runtime").at("pipeline").at("FlowAnalysisHeight")==settings.pipeline.flow_height,"Worker lost motion setting");
        const auto motion=snapshot.at("motion");
        const double wall_ms=snapshot.at("frame_wall_ms").get<double>();
        std::cerr<<Json{{"async_frame",frame},{"motion",motion},{"wall_ms",wall_ms}}.dump()<<std::endl;
        // This worker uses real capture times. Cold model/flow compilation may
        // exceed the one-second discontinuity limit, including after a live
        // analysis-size edit. Preserve that protection and demand valid motion
        // on every subsequent uninterrupted frame, not during cold priming.
        if(frame>1&&frame!=4&&frame!=9&&previous_wall_ms<900)
            phoenix_bundle::require(motion.at("valid").get<bool>(),"Worker did not sustain estimated motion");
        if(frame==9)phoenix_bundle::require(motion.at("reset").get<bool>(),"Worker lost explicit history reset");
        worker_valid_frames+=motion.value("valid",false);
        worker_history_frames+=motion.value("history_usable",false);
        worker.push_back({{"motion",motion},{"wall_ms",wall_ms}});previous_wall_ms=wall_ms;
        auto completed=state->take();
        phoenix_bundle::require(completed.size()==size_t(width)*height*4,"Worker output size differs");
        phoenix_bundle::require(snapshot.at("runtime").at("output_buffer_reused")==bool(frame>1),"Native output buffer was not recycled");
        state->recycle_output(std::move(completed));
    }
    phoenix_bundle::require(worker_valid_frames>=6&&worker_history_frames>=3,"Worker never reached sustained temporal processing");
    state->stop();wait(phoenix_compat::Phase::Stopped);report["async_worker_motion"]=worker;
    report["passed"]=true;report["valid_motion_frames"]=valid_frames;
    report["history_frames"]=history_frames;report["clean_retirement"]=true;
    std::cout<<report.dump(2)<<std::endl;return 0;
} catch(const std::exception& error) {std::cerr<<error.what()<<std::endl;return 1;}

int wmain(int argc,wchar_t** argv) {
    // Match HostClient's fresh x86 worker, rather than borrowing the executable
    // entry thread's loader/TLS context. This is how the qualified bridge runs.
    phoenix_wow64_tests::OpenGlContext graphics;graphics.create();
    int result=1;std::atomic<bool> finished=false;
    std::thread worker([&]{result=validate(argc,argv);finished=true;});
    while(!finished) {
        graphics.render();
        MSG message{};while(PeekMessageW(&message,nullptr,0,0,PM_REMOVE))DispatchMessageW(&message);
        Sleep(5);
    }
    worker.join();graphics.render();return result;
}
