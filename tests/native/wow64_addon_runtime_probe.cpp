#include <windows.h>
#include <DirectXPackedVector.h>
#include "../../src/integration/compatibility/host_client.h"
#include <iostream>
#include <GL/gl.h>
#include <atomic>

// Exercise native-worker startup/retirement while an x86 OpenGL ICD is live.
// This isolates mixed-bitness driver teardown; it is not swapchain acceptance.
class OpenGlContext {
    HWND window_=nullptr;
    HDC dc_=nullptr;
    HGLRC context_=nullptr;
    std::array<void*,64> thread_slots_{};
  public:
    ~OpenGlContext() {
        // Preserve test cleanup if an unqualified native loader cleared an
        // existing x86 slot. Never use this restoration in the actual addon.
        for(unsigned i=0;i<thread_slots_.size();++i)
            if(thread_slots_[i]&&!TlsGetValue(i))TlsSetValue(i,thread_slots_[i]);
        if(context_) {wglMakeCurrent(nullptr,nullptr);wglDeleteContext(context_);}
        if(dc_)ReleaseDC(window_,dc_);
        if(window_)DestroyWindow(window_);
    }
    void create() {
        WNDCLASSW type{};type.style=CS_OWNDC;type.lpfnWndProc=DefWindowProcW;
        type.hInstance=GetModuleHandleW(nullptr);type.lpszClassName=L"PhoenixNativeGlProbe";
        phoenix_bundle::require(RegisterClassW(&type)!=0,"OpenGL probe class failed");
        window_=CreateWindowW(type.lpszClassName,L"Phoenix native OpenGL probe",WS_POPUP,
            0,0,64,64,nullptr,nullptr,type.hInstance,nullptr);
        phoenix_bundle::require(window_!=nullptr,"OpenGL probe window failed");
        dc_=GetDC(window_);
        PIXELFORMATDESCRIPTOR format{};format.nSize=sizeof(format);format.nVersion=1;
        format.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER;
        format.iPixelType=PFD_TYPE_RGBA;format.cColorBits=24;
        const auto index=ChoosePixelFormat(dc_,&format);
        phoenix_bundle::require(index&&SetPixelFormat(dc_,index,&format),"OpenGL pixel format failed");
        context_=wglCreateContext(dc_);
        phoenix_bundle::require(context_&&wglMakeCurrent(dc_,context_),"OpenGL context failed");
        const auto renderer=reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        phoenix_bundle::require(renderer&&strstr(renderer,"NVIDIA"),"Probe needs the NVIDIA x86 OpenGL ICD");
        std::cout<<nlohmann::json{{"opengl_renderer",renderer}}.dump()<<std::endl;
        for(unsigned i=0;i<thread_slots_.size();++i)thread_slots_[i]=TlsGetValue(i);
    }
    void render() {
        for(unsigned i=0;i<thread_slots_.size();++i)
            if(thread_slots_[i]&&!TlsGetValue(i)) {
                std::cout<<nlohmann::json{{"cleared_x86_tls_slot",i}}.dump()<<std::endl;
                throw std::runtime_error("Native worker cleared an existing x86 TLS slot "+std::to_string(i));
            }
        glViewport(0,0,64,64);glClearColor(.25f,.5f,.75f,1.f);glClear(GL_COLOR_BUFFER_BIT);
        unsigned char pixel[3]{};glReadPixels(16,16,1,1,GL_RGB,GL_UNSIGNED_BYTE,pixel);
        phoenix_bundle::require(glGetError()==GL_NO_ERROR&&pixel[0]>=62&&pixel[0]<=66&&
            pixel[1]>=126&&pixel[1]<=130&&pixel[2]>=189&&pixel[2]<=193,"OpenGL probe pixels changed");
        phoenix_bundle::require(SwapBuffers(dc_)!=0,"OpenGL probe swap failed");
    }
};

int validate(int argc,wchar_t** argv) {
    using namespace phoenix_wow64;
    using Json=nlohmann::json;
    try {
        phoenix_bundle::require(argc==3||(argc==4&&!wcscmp(argv[3],L"--opengl")),"runtime root and weight source required");
        const auto root=std::filesystem::absolute(argv[1]);
        const auto source=phoenix_model::WeightSource::config(phoenix_bundle::json_file(argv[2]));
        constexpr unsigned width=640,height=384;
        std::vector<uint8_t> pixels(width*height*4);
        for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) {
            const size_t i=(size_t(y)*width+x)*4;
            pixels[i]=uint8_t(x*255/(width-1));pixels[i+1]=uint8_t(y*255/(height-1));
            pixels[i+2]=uint8_t((x/17+y/13)%2?192:64);pixels[i+3]=255;
        }
        phoenix_compat::Settings settings;
        Json report{{"schema","phoenix-wow64-addon-client/1"},{"passed",false},
            {"caller_bits",sizeof(void*)*8},{"inference_pid",GetCurrentProcessId()},{"helper_pid",0},
            {"retail_games_launched",false},{"game_presentation_validated",false},{"routes",Json::array()}};
        std::vector<float> reference;
        auto changed=[](const auto& a,const auto& b) {
            size_t count=0;for(size_t i=0;i<a.size();++i) if(a[i]!=b[i]) ++count;return count;
        };
        for(bool driver:{true,false}) {
            std::cout<<Json{{"starting_route",driver?3:2}}.dump()<<std::endl;
            FrameSession session(root,root/"native",source,width,height,settings,driver);
            auto first=session.evaluate(pixels,settings,3);
            if(driver) reference=first;
            else phoenix_bundle::require(first==reference,"Packaged routes #2 and #3 differ");
            auto tuning=settings;tuning.controls.enabled=true;tuning.controls.tone=2;tuning.controls.structure=0;
            auto tuned=session.evaluate(pixels,tuning,3);
            const auto controls_changed=changed(first,tuned);
            phoenix_bundle::require(controls_changed>1000,"Live controls did not change output");
            auto restored=session.evaluate(pixels,settings,3);
            phoenix_bundle::require(restored==first,"Restoring live controls did not restore output");
            auto second=session.evaluate(pixels,settings,3,&first);
            phoenix_bundle::require(changed(first,second)>1000,"Second pass did not change output");
            Json row{{"route",driver?3:2},{"backend","phoenix"},{"frames",4},
                {"controls_changed_values",controls_changed},{"restored_exactly",true},
                {"refinement_changed_values",changed(first,second)},{"gpu_ms",session.gpu_ms()}};
            if(!driver) {
                auto nv=session.evaluate(pixels,settings,1);
                phoenix_bundle::require(nv.size()==first.size(),"NVIDIA DLSS5 output extent mismatch");
                auto nv_tuned=session.evaluate(pixels,tuning,1);
                const auto nv_status=session.nvidia_status();
                phoenix_bundle::require(nv_status.at("ready")&&nv_status.at("creates")==1&&nv_status.at("evaluates")==2,
                    "Native NVIDIA DLSS5 status did not report real feature evaluations");
                phoenix_bundle::require(changed(nv,nv_tuned)>1000,"NVIDIA DLSS5 controls did not change output");
                auto switched=session.evaluate(pixels,settings,3);
                phoenix_bundle::require(switched==first,"NVIDIA DLSS5/Phoenix DLSS5 switching did not restore output");
                auto comparison=settings;comparison.mode=phoenix_addon::phoenix::split;comparison.comparison=1;
                comparison.source_a=1;comparison.source_b=2;comparison.divider=.5f;
                auto composed=phoenix_compat::portable_compose(pixels,first,width,comparison,&nv);
                phoenix_bundle::require(composed.size()==pixels.size(),"Comparison composition extent mismatch");
                row["nvidia_frames"]=2;row["nvidia_controls_changed_values"]=changed(nv,nv_tuned);
                row["nvidia_phoenix_switch_restored_exactly"]=true;row["comparison_composed"]=true;
            }
            session.retire();row["clean_retirement"]=true;
            report["routes"].push_back(row);
            std::cout<<row.dump()<<std::endl;
        }
        auto wait=[](const auto& state,phoenix_compat::Phase phase) {
            const auto deadline=GetTickCount64()+30000;
            while(state->phase()!=phase&&state->phase()!=phoenix_compat::Phase::Failed&&GetTickCount64()<deadline) Sleep(1);
            if(state->phase()!=phase) throw std::runtime_error("Addon worker: "+state->error());
        };
        for(unsigned worker_case=0;worker_case<4;++worker_case) {
            const bool driver=worker_case==1;
            auto worker_settings=settings;
            if(worker_case==2) worker_settings.mode=phoenix_addon::phoenix::nvidia;
            if(worker_case==3) {worker_settings.mode=phoenix_addon::phoenix::split;worker_settings.comparison=1;}
            FeedBuild build{};
            build.width=build.output_width=width;build.height=build.output_height=height;build.cpu_bridge=true;
            auto state=std::make_shared<phoenix_compat::HostState>(worker_settings);
            state->start(nullptr,source,worker_settings,build,std::nullopt,phoenix_compat::NativeOptions{root,driver});
            wait(state,phoenix_compat::Phase::Ready);
            phoenix_bundle::require(state->process_id()==GetCurrentProcessId(),"Addon worker launched a helper");
            phoenix_bundle::require(state->submit(1,true,pixels),"Addon worker refused input");
            wait(state,phoenix_compat::Phase::Finished);const auto first=state->take();
            auto tuned=worker_settings;tuned.controls.enabled=true;tuned.controls.tone=2;tuned.controls.structure=0;
            phoenix_bundle::require(state->update_tuning(tuned)&&state->submit(2,true,pixels,2),"Addon worker rejected tuning/iterations");
            wait(state,phoenix_compat::Phase::Finished);const auto second=state->take();
            phoenix_bundle::require(changed(first,second)>1000&&state->tuning_current_output(),"Addon worker did not present updated tuning");
            const auto snapshot=state->ui_snapshot();
            phoenix_bundle::require(snapshot.at("inference_pid")==GetCurrentProcessId()&&snapshot.at("helper_pid")==0,
                "Addon worker has incorrect process provenance");
            const auto stopped=GetTickCount64();state->stop();
            phoenix_bundle::require(GetTickCount64()-stopped<20,"Stopping addon worker blocked caller");
            wait(state,phoenix_compat::Phase::Stopped);
            report["async_workers"].push_back(Json{{"route",driver?3:2},{"passed",true},{"live_tuning",true},
                {"mode",worker_settings.mode},{"iterations",2},{"nonblocking_stop",true},{"processing_location",snapshot.at("processing_location")}});
        }
        report["passed"]=true;std::cout<<report.dump()<<std::endl;return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<std::endl;return 2;}
}

int wmain(int argc,wchar_t** argv) {
    // Real add-ons first load native images from a fresh x86 processing worker.
    // Main-thread priming can hide missing per-thread native loader state.
    OpenGlContext graphics;
    const bool opengl=argc==4&&!wcscmp(argv[3],L"--opengl");
    try {if(opengl)graphics.create();}
    catch(const std::exception& error) {std::cerr<<error.what()<<std::endl;return 2;}
    int result=2;
    std::atomic<bool> finished=false;
    std::thread worker([&]{result=validate(argc,argv);finished=true;});
    std::string graphics_error;
    while(opengl&&!finished) {
        if(graphics_error.empty()) {
            try {graphics.render();}
            catch(const std::exception& error) {graphics_error=error.what();}
        }
        MSG message{};while(PeekMessageW(&message,nullptr,0,0,PM_REMOVE))DispatchMessageW(&message);
        Sleep(5);
    }
    worker.join();
    if(opengl&&graphics_error.empty()) {
        try {graphics.render();}
        catch(const std::exception& error) {graphics_error=error.what();}
    }
    if(!graphics_error.empty()){std::cerr<<graphics_error<<std::endl;return 2;}
    return result;
}
