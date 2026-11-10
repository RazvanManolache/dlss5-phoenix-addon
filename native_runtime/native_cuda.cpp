#define PHOENIX_RUNTIME_IMPLEMENTATION
#include "phoenix.h"
#include "phoenix_linux.h"
#include "native_cuda_executor.h"
#include "advanced_frame.h"
#ifdef _WIN32
#include "native_cuda_d3d12.h"
#endif
#include <mutex>

namespace {
using Json = nlohmann::json;
thread_local std::string last_error;
struct Session {
    std::mutex mutex;
    Json configuration, signature;
    std::unique_ptr<phoenix_native_cuda::Executor> executor;
    // Children borrow this session's immutable weights/modules, but own history/workspace.
    std::vector<std::unique_ptr<phoenix_native_cuda::Executor>> passes;
#ifdef _WIN32
    std::unique_ptr<phoenix_native_cuda::D3D12Bridge> d3d12;
#endif
    phoenix_cpp::NetworkSchedule schedule;
    std::vector<float> previous_distortion;
    uint64_t frames=0;
    Session(const Json& config,unsigned width,unsigned height)
        : configuration(config),executor(std::make_unique<phoenix_native_cuda::Executor>(config,width,height)) {
        executor->status["capabilities"]={{"granular_controls",true},{"temporal_reuse",true},{"frame_api_versions",{1,2,3}}};
    }
};
size_t json_result(const Json& value,char* destination,size_t capacity) {
    const auto text=value.dump();const size_t bytes=text.size()+1;
    if(destination&&capacity>=bytes)std::memcpy(destination,text.c_str(),bytes);
    return bytes;
}
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
struct Request {
    std::vector<float> color,motion,depth,validity,mask,ui,alpha,backbuffer,distortion;
    Json metadata;
    size_t output_stride{};
};
void validate_metadata(const Json& metadata){
    require(metadata.is_object(),"Native CUDA metadata must be an object");
    const std::vector<std::string> allowed{"reset","has_motion","has_depth","has_history_validity","has_control_mask","controls","transformer_blocks","repeats","passes","adaptive_budget","presentation","reversed_depth","motion_from_frame","output_stage"};
    for(const auto& field:metadata.items())require(std::find(allowed.begin(),allowed.end(),field.key())!=allowed.end(),"Unknown native CUDA frame metadata; frame services use the shared pipeline SDK");
}
Request read_frame(Session& session,const phoenix_frame_v2& frame,const phoenix_frame_v3* advanced) {
    const auto w=session.executor->width,h=session.executor->height;
    require(!frame.reserved&&(frame.control_profile==PHOENIX_CONTROL_NONE||frame.control_profile==PHOENIX_CONTROL_EXPERIMENTAL_RGB_1),"Invalid native CUDA frame fields");
    Request r;r.output_stride=phoenix_cpp::plane_stride(frame.output,w,h,3);
    r.metadata=frame.metadata_json?Json::parse(frame.metadata_json):Json::object();
    validate_metadata(r.metadata);
    r.color=phoenix_cpp::copy_plane(frame.color,w,h,3,true);
    r.motion=phoenix_cpp::copy_plane(frame.motion,w,h,2,false);
    r.depth=phoenix_cpp::copy_plane(frame.depth,w,h,1,false);
    r.validity=phoenix_cpp::copy_plane(frame.validity,w,h,1,false);
    r.mask=phoenix_cpp::copy_plane(frame.control_mask,w,h,4,false);
    require((frame.control_profile==PHOENIX_CONTROL_EXPERIMENTAL_RGB_1)==!r.mask.empty(),"Control mask requires the explicit RGB profile");
    for(auto entry:{std::pair{"has_motion",!r.motion.empty()},std::pair{"has_depth",!r.depth.empty()},std::pair{"has_history_validity",!r.validity.empty()},std::pair{"has_control_mask",!r.mask.empty()}}) {
        if(r.metadata.contains(entry.first))require(r.metadata.at(entry.first).is_boolean()&&r.metadata.at(entry.first).get<bool>()==entry.second,"Declared native plane differs from bounded plane");
        r.metadata[entry.first]=entry.second;
    }
    require(r.validity.empty()||!r.motion.empty(),"History validity requires motion");
    for(float value:r.color)require(std::isfinite(value)&&value>=0&&value<=1,"Native color requires finite SDR proxy values in 0..1");
    for(float value:r.motion)require(std::isfinite(value),"Nonfinite native motion");
    for(float value:r.depth)require(std::isfinite(value),"Nonfinite native depth");
    for(float value:r.validity)require(value==0||value==1,"Native validity must be binary");
    phoenix_cpp::validate_control_mask(r.mask.empty()?nullptr:r.mask.data(),size_t(w)*h);
    if(advanced){
        r.ui=phoenix_cpp::copy_plane(advanced->ui,w,h,4,false);r.alpha=phoenix_cpp::copy_plane(advanced->ui_alpha,w,h,1,false);
        r.backbuffer=phoenix_cpp::copy_plane(advanced->backbuffer,w,h,3,false);r.distortion=phoenix_cpp::copy_plane(advanced->distortion,w,h,4,false);
        for(const auto* plane:{&r.ui,&r.alpha,&r.backbuffer})for(float value:*plane)require(std::isfinite(value)&&value>=0&&value<=1,"Invalid native display plane");
        for(size_t p=0;p<r.ui.size()/4;++p){const float a=r.ui[p*4+3];require(r.alpha.empty()||r.alpha[p]==a,"Conflicting UI coverage");
            for(unsigned c=0;c<3;++c)require(r.ui[p*4+c]<=a,"UI must be premultiplied");}
        for(float value:r.distortion)require(std::isfinite(value)&&std::fabs(value)<=1,"Distortion must contain normalized finite displacements");
    }
    return r;
}
struct EvaluationPlan {
    Json controls_json,signature;
    phoenix_cpp::Controls controls;
    phoenix_cpp::NetworkSchedule schedule;
    std::string output_stage;
    bool reversed{},reset{};
};
EvaluationPlan plan_frame(Session& session,const Json& metadata,bool motion,bool mask,bool distortion){
    EvaluationPlan plan;
    Json controls_json={{"local_tone",session.configuration.value("local_tone",1.f)},
        {"local_structure",session.configuration.value("local_structure",1.f)},
        {"skin_structure",session.configuration.value("skin_structure",-1.f)},
        {"auto_mask",session.configuration.value("auto_mask",true)}};
    if(metadata.contains("controls")){
        require(metadata.at("controls").is_object(),"Frame controls must be an object");
        for(const auto& field:metadata.at("controls").items()){
            require(field.key()=="intensity"||field.key()=="style"||field.key()=="local_tone"||field.key()=="local_structure"||field.key()=="skin_structure"||field.key()=="auto_mask","Unsupported native appearance control");
            controls_json[field.key()]=field.value();}
    }
    plan.controls_json=controls_json;
    const auto controls=plan.controls=phoenix_cpp::Controls::from(controls_json);
    if(controls.style)require(phoenix_cpp::has_recovered_style_table(session.executor->status.at("model_resource_sha256").get<std::string>()),"Selected model has no verified style preset table");
    const auto output_stage=plan.output_stage=metadata.value("output_stage",std::string("presented"));
    require(output_stage=="presented"||output_stage=="neural_only","Unknown native output stage");
    auto schedule=session.schedule;
    schedule.prepare(metadata,session.executor->status.at("model_resource_sha256").get<std::string>(),true);
    const bool supplied_reset=metadata.value("reset",false);
    const bool reversed=plan.reversed=metadata.value("reversed_depth",false);
    plan.signature={schedule.repeats,schedule.passes,controls.tone,controls.structure,controls.skin,controls.auto_mask,reversed,motion,mask,distortion};
    plan.reset=supplied_reset||plan.signature!=session.signature;plan.schedule=schedule;
    return plan;
}
void evaluate(Session& session,Request& r,const phoenix_output_f32& destination){
    auto plan=plan_frame(session,r.metadata,!r.motion.empty(),!r.mask.empty(),!r.distortion.empty());
    const auto& controls=plan.controls;auto schedule=plan.schedule;const auto& controls_json=plan.controls_json;
    const auto& output_stage=plan.output_stage;const auto& signature=plan.signature;const bool reset=plan.reset,reversed=plan.reversed;
    const unsigned w=session.executor->width,h=session.executor->height;
    phoenix_cpp::CanonicalFrame canonical;
    const auto data=[](const std::vector<float>& v){return v.empty()?nullptr:v.data();};
    const float* color=r.color.data();const float* motion=data(r.motion);const float* depth=data(r.depth);const float* validity=data(r.validity);const float* mask=data(r.mask);
    if(!r.distortion.empty()){
        canonical.prepare(color,motion,depth,validity,mask,r.distortion.data(),reset?nullptr:data(session.previous_distortion),w,h);
        color=canonical.color.data();motion=data(canonical.motion);depth=data(canonical.depth);validity=data(canonical.validity);mask=data(canonical.mask);
    }
    while(session.passes.size()+1<schedule.passes)session.passes.push_back(std::make_unique<phoenix_native_cuda::Executor>(session.configuration,w,h,session.executor.get()));
    std::vector<phoenix_native_cuda::Executor*> used{session.executor.get()};
    for(unsigned pass=1;pass<schedule.passes;++pass)used.push_back(session.passes[pass-1].get());
    const std::vector<float>* output=nullptr;
    std::vector<phoenix_cpp::Event> events;
    for(auto* executor:used){
        executor->prepare_frame(reset,motion,depth,validity,mask,controls,reversed,schedule.repeats);
        output=&executor->run(color);color=output->data();
        events.push_back({0,0,executor->status.at("operator_ms").get<double>(),true,0});
    }
    std::vector<float> presented=*output;
    if(!r.distortion.empty())presented=phoenix_cpp::warp_plane(presented.data(),r.distortion.data(),w,h,3,0);
    if(output_stage!="neural_only"){
        phoenix_cpp::DisplayInputs display{data(r.mask),data(r.ui),data(r.alpha),data(r.backbuffer)};
        for(size_t p=0;p<size_t(w)*h;++p){phoenix_cpp::RGB original,neural;
            for(unsigned c=0;c<3;++c){original[c]=r.color[p*3+c];neural[c]=presented[p*3+c];}
            const auto pixel=phoenix_cpp::compose_display(original,neural,p,controls.intensity,controls.style,controls.tone,display);
            for(unsigned c=0;c<3;++c)presented[p*3+c]=pixel[c];}
    }
    for(float value:presented)require(std::isfinite(value),"Nonfinite native presentation output");
    for(auto* executor:used)executor->commit();
    schedule.complete(events);session.schedule=schedule;session.signature=signature;session.previous_distortion=std::move(r.distortion);
    ++session.frames;session.executor->status["frames"]=session.frames;
    session.executor->status["network_schedule"]=schedule.status();
    session.executor->status["passes"]=schedule.passes;
    session.executor->status["completed_controls"]=controls_json;
    session.executor->status["pixel_transport"]="bounded-rgb32f";
    session.executor->status["neural_host_transfer_bytes"]=uint64_t(w)*h*schedule.passes*(24+(motion?8:0)+(depth?4:0)+(mask?16:0)+(motion&&depth&&validity?8:0));
    if(session.configuration.value("diagnostics",false)){
        Json samples={{"input",Json::array()},{"neural",Json::array()},{"presented",Json::array()}};
        for(unsigned y:{h/4,h/2,h*3/4})for(unsigned x:{w/4,w/2,w*3/4})for(unsigned c=0;c<3;++c){
            const auto offset=(size_t(y)*w+x)*3+c;
            samples["input"].push_back(r.color[offset]);samples["neural"].push_back((*output)[offset]);samples["presented"].push_back(presented[offset]);
        }
        session.executor->status["completed_rgb_samples"]=std::move(samples);
    }
    auto* bytes=reinterpret_cast<uint8_t*>(destination.data);
    for(unsigned y=0;y<h;++y)std::memcpy(bytes+size_t(y)*r.output_stride,presented.data()+size_t(y)*w*3,size_t(w)*3*sizeof(float));
}
#ifdef _WIN32
int gpu_frame(Session& session,const phoenix_cuda_d3d12_frame& frame){
    require(frame.queue&&frame.color&&frame.output&&frame.color!=frame.output&&frame.motion!=frame.output&&frame.color!=frame.motion,"Invalid D3D12 frame resources or alias");
    require(std::isfinite(frame.motion_scale_x)&&std::isfinite(frame.motion_scale_y),"Nonfinite D3D12 motion scales");
    auto metadata=frame.metadata_json?Json::parse(frame.metadata_json):Json::object();validate_metadata(metadata);
    for(auto field:{std::pair{"has_motion",frame.motion!=nullptr},std::pair{"has_depth",false},std::pair{"has_history_validity",false},std::pair{"has_control_mask",false}})
        if(metadata.contains(field.first))require(metadata.at(field.first).is_boolean()&&metadata.at(field.first).get<bool>()==field.second,"D3D12 metadata plane mismatch");
    auto plan=plan_frame(session,metadata,frame.motion!=nullptr,false,false);
    require(plan.schedule.passes==1,"D3D12 neural transport accepts one pass per stage");
    if(!session.configuration.value("d3d12_interop",true))return 1;
    auto& executor=*session.executor;executor.activate();
    if(!session.d3d12)session.d3d12=std::make_unique<phoenix_native_cuda::D3D12Bridge>(executor.device_ordinal(),executor.width,executor.height,frame.queue);
    auto& bridge=*session.d3d12;bridge.validate(frame);
    // A host/device transport transition changes the guide representation and
    // starts a fresh stage; subsequent device frames retain their GPU history.
    plan.signature.push_back("d3d12");
    plan.reset=metadata.value("reset",false)||plan.signature!=session.signature;
    const auto before=executor.status;
    try{
        bridge.prepare(frame,executor.stream());
        executor.prepare_frame(plan.reset,nullptr,nullptr,nullptr,nullptr,plan.controls,plan.reversed,plan.schedule.repeats,
            frame.motion?executor.device_motion():0,bridge.output());
        const bool appearance=plan.output_stage!="neural_only";
        uint64_t signal{};
        executor.run_external(bridge.color(),frame.motion?bridge.motion():0,bridge.output(),frame.motion_scale_x,frame.motion_scale_y,[&]{
            executor.enqueue_external_finish(bridge.color(),bridge.alpha(),bridge.output(),
                appearance?plan.controls.intensity:1.f,appearance?plan.controls.style:0,plan.controls.tone);
            signal=bridge.signal_cuda(executor.stream());
        });
        bridge.finish_completed(frame,signal);
        executor.mark_committed();
    }catch(const std::invalid_argument&){executor.status=before;throw;}
    catch(...){executor.poison();throw;}
    plan.schedule.complete({{0,0,executor.status.at("operator_ms").get<double>(),true,0}});
    session.schedule=plan.schedule;session.signature=std::move(plan.signature);session.previous_distortion.clear();
    executor.status["frames"]=++session.frames;executor.status["network_schedule"]=plan.schedule.status();executor.status["passes"]=1;
    executor.status["completed_controls"]=plan.controls_json;executor.status["gpu_shared_resource_bytes"]=bridge.bytes();executor.status.erase("completed_rgb_samples");
    return 0;
}
#endif
int bounded(void* handle,const phoenix_frame_v2* frame,const phoenix_frame_v3* advanced){
    try{
        phoenix_native_cuda::CurrentContextGuard context;
        require(handle&&frame&&frame->struct_bytes==(advanced?sizeof(phoenix_frame_v3):sizeof(phoenix_frame_v2))&&frame->version==(advanced?3:2),"Invalid native CUDA frame header");
        auto& session=*static_cast<Session*>(handle);std::lock_guard guard(session.mutex);
        auto request=read_frame(session,*frame,advanced);evaluate(session,request,frame->output);last_error.clear();return 0;
    }catch(const std::exception& error){last_error=error.what();return -1;}
}
}
extern "C" void* phoenix_runtime_create(const char* configuration,unsigned width,unsigned height){
    try{phoenix_native_cuda::CurrentContextGuard context;require(configuration,"Missing native CUDA configuration");
        auto result=std::make_unique<Session>(Json::parse(configuration),width,height);last_error.clear();return result.release();
    }catch(const std::exception& error){last_error=error.what();return nullptr;}
}
extern "C" void phoenix_runtime_destroy(void* session){try{phoenix_native_cuda::CurrentContextGuard context;delete static_cast<Session*>(session);}catch(...){}}
extern "C" const char* phoenix_runtime_error(){return last_error.c_str();}
extern "C" int phoenix_runtime_frame_v2(void* handle,const phoenix_frame_v2* frame){return bounded(handle,frame,nullptr);}
extern "C" int phoenix_runtime_frame_v3(void* handle,const phoenix_frame_v3* frame){return bounded(handle,frame?&frame->base:nullptr,frame);}
#ifdef _WIN32
extern "C" PHOENIX_CUDA_D3D12_API int phoenix_runtime_frame_d3d12(void* handle,const phoenix_cuda_d3d12_frame* frame){
    try{
        phoenix_native_cuda::CurrentContextGuard context;
        require(handle&&frame&&frame->struct_bytes==sizeof(*frame)&&frame->version==1&&!frame->reserved[0]&&!frame->reserved[1],"Invalid native D3D12 frame header");
        auto& session=*static_cast<Session*>(handle);std::lock_guard guard(session.mutex);
        const auto result=gpu_frame(session,*frame);last_error.clear();return result;
    }catch(const phoenix_native_cuda::InteropUnavailable& error){last_error=error.what();return 1;}
    catch(const std::exception& error){last_error=error.what();return -1;}
}
#endif
extern "C" int phoenix_runtime_frame(void* handle,const float* color,const float* motion,const float* depth,const float* validity,const char* text,float* output){
    try{
        require(handle&&color&&output,"Missing native CUDA frame");auto& session=*static_cast<Session*>(handle);
        const auto metadata=text?Json::parse(text):Json::object();const unsigned w=session.executor->width,h=session.executor->height;
        auto plane=[&](const float* p,unsigned channels,bool enabled){return enabled?phoenix_plane_f32{p,uint64_t(w)*h*channels*4,0,w,h,channels,0}:phoenix_plane_f32{};};
        phoenix_frame_v2 frame{};frame.struct_bytes=sizeof(frame);frame.version=2;
        frame.color=plane(color,3,true);frame.motion=plane(motion,2,metadata.value("has_motion",false));frame.depth=plane(depth,1,metadata.value("has_depth",false));
        frame.validity=plane(validity,1,metadata.value("has_history_validity",false));frame.output={output,uint64_t(w)*h*3*4,0,w,h,3,0};frame.metadata_json=text;
        return bounded(handle,&frame,nullptr);
    }catch(const std::exception& error){last_error=error.what();return -1;}
}
extern "C" size_t phoenix_runtime_status(void* handle,char* destination,size_t capacity){
    try{require(handle,"Missing native CUDA session");auto& session=*static_cast<Session*>(handle);std::lock_guard guard(session.mutex);return json_result(session.executor->status,destination,capacity);}
    catch(const std::exception& error){last_error=error.what();return 0;}
}
extern "C" size_t phoenix_runtime_capabilities(char* destination,size_t capacity){
    return json_result({{"schema","phoenix-native-cuda-capabilities/2"},{"host_frame_api_versions",{1,2,3}},
        {"frame_contract","temporal-rgb/1"},{"architectures",{"sm_120"}},{"motion",true},{"depth",true},{"history_validity",true},
        {"temporal_reuse",true},{"advanced_controls",true},{"granular_controls",true},{"model_weights_included",false}},destination,capacity);
}
extern "C" int phoenix_runtime_replace_model(void* handle,const char* source){
    try{
        phoenix_native_cuda::CurrentContextGuard context;require(handle&&source,"Missing native model replacement");
        auto& session=*static_cast<Session*>(handle);std::lock_guard guard(session.mutex);
        auto configuration=session.configuration;configuration["weight_source"]=Json::parse(source);
        auto replacement=std::make_unique<phoenix_native_cuda::Executor>(configuration,session.executor->width,session.executor->height);
#ifdef _WIN32
        session.d3d12.reset(); // Retire imports while the old executor's stream is still alive.
#endif
        session.passes.clear();session.executor.swap(replacement);session.configuration=std::move(configuration);
        session.signature=nullptr;session.previous_distortion.clear();session.schedule={};session.frames=0;
        session.executor->status["capabilities"]={{"granular_controls",true},{"temporal_reuse",true},{"frame_api_versions",{1,2,3}}};
        last_error.clear();return 0;
    }catch(const std::exception& error){last_error=error.what();return -1;}
}
extern "C" PHOENIX_RGBA8_API int phoenix_linux_frame_rgba8(void* handle,const phoenix_rgba8_frame* frame){
    try{
        phoenix_native_cuda::CurrentContextGuard context;
        require(handle&&frame&&frame->struct_bytes==sizeof(*frame)&&frame->version==1&&frame->format<=1&&frame->flags==1&&frame->data,"Invalid native RGBA8 frame header");
        auto& session=*static_cast<Session*>(handle);std::lock_guard guard(session.mutex);
        const auto w=session.executor->width,h=session.executor->height;const uint64_t row=uint64_t(w)*4,stride=frame->row_stride_bytes?frame->row_stride_bytes:row;
        require(frame->width==w&&frame->height==h&&stride>=row&&stride<=(UINT64_MAX-row)/h,"Invalid native RGBA8 extent/stride");
        const uint64_t required=stride*(h-1)+row;
        require(required<=frame->capacity_bytes&&required<=SIZE_MAX&&required<=UINTPTR_MAX-reinterpret_cast<uintptr_t>(frame->data),"Insufficient native RGBA8 capacity");
        Json controls={{"local_tone",session.configuration.value("local_tone",1.f)},{"local_structure",session.configuration.value("local_structure",1.f)},
            {"skin_structure",session.configuration.value("skin_structure",-1.f)},{"auto_mask",session.configuration.value("auto_mask",true)}};
        session.executor->prepare_frame(true,nullptr,nullptr,nullptr,nullptr,phoenix_cpp::Controls::from(controls),false,{1,1,1,1,1,1,1,1});
        const auto* pixels=session.executor->run_rgba8(static_cast<const uint8_t*>(frame->data),size_t(stride),frame->format==1);
        session.executor->commit();session.signature=nullptr;session.previous_distortion.clear();session.executor->status["frames"]=++session.frames;
        for(unsigned y=0;y<h;++y)std::memcpy(static_cast<uint8_t*>(frame->data)+size_t(y)*size_t(stride),pixels+size_t(y)*w*4,size_t(w)*4);
        last_error.clear();return 0;
    }catch(const std::exception& error){last_error=error.what();return -1;}
}
extern "C" PHOENIX_RGBA8_API int phoenix_runtime_frame_rgba8(void* handle,const phoenix_rgba8_frame* frame){return phoenix_linux_frame_rgba8(handle,frame);}
