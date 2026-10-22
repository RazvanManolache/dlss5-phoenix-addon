// Included after PhoenixNativeSession is complete. Every network pass/branch
// is an owned session; histories never bleed across passes or input streams.
struct PhoenixNativeSession::Graph {
    PhoenixNativeSession& owner;
    struct Image {ComPtr<ID3D12Resource> value;unsigned w=0,h=0;DXGI_FORMAT format=DXGI_FORMAT_UNKNOWN;};
    std::map<std::string,Image> images;
    std::map<std::string,std::unique_ptr<PhoenixNativeSession>> networks;
    std::unique_ptr<spatial::Pass> operations;
    Json executed=Json::array();
    uint64_t evaluations=0;
    explicit Graph(PhoenixNativeSession& parent):owner(parent){}
    void prepare_neural(const std::string& key,unsigned w,unsigned h,float scale){
        const auto nw=std::max(1u,unsigned(std::lround(w*scale))),nh=std::max(1u,unsigned(std::lround(h*scale)));
        for(unsigned pass=0;pass<owner.tuning.passes;++pass){
            const auto name=key+"/pass"+std::to_string(pass);
            if(networks.contains(name))continue;
            auto config=owner.configuration;config.erase("tuning");
            if(owner.portable)config["_portable_leaf"]=true;
            config["width"]=nw;config["height"]=nh;config["output_width"]=nw;config["output_height"]=nh;config["neural_width"]=nw;config["neural_height"]=nh;
            config["repeats"]=owner.tuning.repeats;config["transformer_blocks"]=owner.tuning.blocks;
            config["pipeline"]={{"PipelineMonitoring",owner.pipeline.monitoring},{"MonitoringSampleInterval",owner.pipeline.sample_interval}};
            networks.emplace(name,std::make_unique<PhoenixNativeSession>(config,owner.device.Get(),owner.queue.Get()));
        }
    }
    ID3D12Resource* image(const std::string& key,unsigned w,unsigned h,DXGI_FORMAT format=DXGI_FORMAT_R16G16B16A16_FLOAT){
        auto& slot=images[key];
        if(!slot.value||slot.w!=w||slot.h!=h||slot.format!=format){
            slot.value=owner.image(w,h,format);slot.w=w;slot.h=h;slot.format=format;
        }
        return slot.value.Get();
    }
    void run(ID3D12Resource* source,D3D12_RESOURCE_STATES state,ID3D12Resource* output,
             unsigned mode=0,ID3D12Resource* other=nullptr,float a=0,float b=1,float c=1,float d=1){
        if(!operations)operations=std::make_unique<spatial::Pass>(owner.device.Get(),phoenix_pipeline::image_ops_shader,owner.accounting);
        const auto desc=output->GetDesc();
        phoenix_pipeline::ImageConstants values{unsigned(desc.Width),desc.Height,mode,0,a,b,c,d,0};
        owner.begin();
        phoenix_native::transition(owner.command.Get(),source,state,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::transition(owner.command.Get(),output,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        operations->record(owner.command.Get(),{source,other,nullptr},output,values);
        phoenix_native::transition(owner.command.Get(),output,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::transition(owner.command.Get(),source,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,state);
        owner.submit();owner.wait(owner.submitted);
    }
    ID3D12Resource* resample(const std::string& key,ID3D12Resource* source,D3D12_RESOURCE_STATES state,unsigned w,unsigned h,bool motion=false){
        auto* target=image(key,w,h,motion?DXGI_FORMAT_R16G16_FLOAT:DXGI_FORMAT_R16G16B16A16_FLOAT);
        run(source,state,target,motion?1:0);return target;
    }
    ID3D12Resource* neural(const std::string& key,ID3D12Resource* source,ID3D12Resource* motion,
                          bool reset,const PhoenixNativeControls& controls,unsigned w,unsigned h,float scale){
        const unsigned nw=std::max(1u,unsigned(std::lround(w*scale))),nh=std::max(1u,unsigned(std::lround(h*scale)));
        auto* input=source;
        if(nw!=w||nh!=h)input=resample(key+"/input",source,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,nw,nh);
        ID3D12Resource* mv=motion;
        if(motion&&(nw!=owner.width||nh!=owner.height))mv=resample(key+"/motion",motion,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,nw,nh,true);
        const auto& tuning=owner.tuning;
        prepare_neural(key,w,h,scale);
        for(unsigned pass=0;pass<tuning.passes;++pass){
            const auto name=key+"/pass"+std::to_string(pass);
            auto& stage=networks.at(name);
            if(stage->pipeline.monitoring!=owner.pipeline.monitoring||stage->pipeline.sample_interval!=owner.pipeline.sample_interval){
                const Json monitor{{"PipelineMonitoring",owner.pipeline.monitoring},{"MonitoringSampleInterval",owner.pipeline.sample_interval}};
                require(phoenix_native_set_pipeline(stage.get(),monitor.dump().c_str())==0,"Cannot update stage monitoring");
            }
            auto* result=image(name+"/output",nw,nh);
            auto options=controls;options.reset=reset||!tuning.temporal;
            if(owner.configuration.contains("tuning")){
                options.local_tone=tuning.tone;options.local_structure=tuning.structure;
                options.skin_structure=tuning.skin;options.auto_mask=tuning.auto_mask;
            }
            if(owner.frame_info_pending){stage->next_frame_info=owner.next_frame_info;stage->frame_info_pending=true;}
            const bool appearance=!owner.spatial_only&&(tuning.intensity!=1||tuning.style!=0);
            PhoenixNativeFrameV2 frame{};
            if(appearance){frame.struct_bytes=sizeof(frame);frame.version=2;frame.controls=options;
                frame.intensity=tuning.intensity;frame.style=tuning.style;}
            stage->frame(input,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,mv,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                result,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,&options,appearance?&frame:nullptr);
            if(!owner.spatial_only){++evaluations;++owner.graph_evaluations;}
            if(stage->gpu_sampled_last_frame){owner.gpu_sampled_last_frame=true;owner.gpu_ms+=stage->gpu_ms;}
            if(tuning.color_strength!=1){
                auto* color=image(name+"/chroma",nw,nh);
                run(result,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,color,4,input,tuning.color_strength);result=color;
            }
            input=result;
        }
        if(nw!=w||nh!=h)input=resample(key+"/restored",input,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,w,h);
        return input;
    }
};

void PhoenixNativeSession::initialize_graph(){
    auto prepared=std::make_shared<Graph>(*this);
    unsigned w=width,h=height;
    for(const auto step:phoenix_pipeline::plan(pipeline,width,height,output_width,output_height)){
        if(step.stage==phoenix_pipeline::Stage::Neural)prepared->prepare_neural("neural",w,h,tuning.neural_scale/100);
        else if(step.stage==phoenix_pipeline::Stage::Fusion){
            if(!phoenix_pipeline::fusion_reuses_neural(pipeline,width,height,output_width,output_height))prepared->prepare_neural("fusion0",w,h,tuning.neural_scale/100);
            for(int layer=1;layer<pipeline.fusion_layers;++layer)
                prepared->prepare_neural("fusion"+std::to_string(layer),w,h,layer==1?pipeline.fusion_second_scale:pipeline.fusion_third_scale);
        }
        else if(step.stage==phoenix_pipeline::Stage::Scale){
            if(pipeline.scaler==3){require(step.width==w*2&&step.height==h*2,"Tiny reconstruction requires 2x output geometry");
                if(!tiny)tiny=std::make_unique<phoenix_pipeline::TinyUpscaler>(device.Get(),w,h,tiny_model_path);}
            if(pipeline.scaler==1){require(step.width>=w&&step.height>=h&&step.width<=w*2&&step.height<=h*2,"NIS admits 1x to 2x output per axis");
                if(!nis)nis=std::make_unique<phoenix_pipeline::NisScaler>(device.Get());}
            w=step.width;h=step.height;
        }
    }
    if(!prepared->networks.empty())packed_weights_sha256=prepared->networks.begin()->second->packed_weights_sha256;
    graph=std::move(prepared);
}

void PhoenixNativeSession::frame_graph(ID3D12Resource* color,D3D12_RESOURCE_STATES color_state,
    ID3D12Resource* motion,D3D12_RESOURCE_STATES motion_state,ID3D12Resource* output,
    D3D12_RESOURCE_STATES output_state,const PhoenixNativeControls* supplied,const PhoenixNativeFrameV2* extended){
    validate(color,DXGI_FORMAT_R16G16B16A16_FLOAT);validate(output,DXGI_FORMAT_R16G16B16A16_FLOAT,true);
    if(motion)validate(motion,DXGI_FORMAT_R16G16_FLOAT);
    require(color!=output&&motion!=output&&color!=motion,"Pipeline texture alias rejected");
    require(!extended||(!extended->control_mask.resource&&!extended->ui.resource&&!extended->ui_alpha.resource&&
        !extended->backbuffer.resource&&!extended->distortion.resource),"Optional frame planes require a dedicated chain stage");
    PhoenixNativeControls controls;phoenix_native_default_controls(&controls);if(supplied)controls=*supplied;
    require(controls.struct_bytes==sizeof(controls),"Pipeline control structure mismatch");
    const auto schedule=phoenix_pipeline::plan(pipeline,width,height,output_width,output_height);
    const bool monitored=pipeline.monitoring!=0;
    const bool measured=monitored||neural_budget_enabled();
    using Clock=std::chrono::steady_clock;
    const auto started=measured?Clock::now():Clock::time_point{};
    auto mark=started;std::array<double,5> stages{};
    auto elapsed=[&](unsigned stage){if(measured){const auto now=Clock::now();stages[stage]+=std::chrono::duration<double,std::milli>(now-mark).count();mark=now;}};
    gpu_sampled_last_frame=false;gpu_ms=0;
    retained_optional.clear();
    for(auto* borrowed:{color,motion,output})if(borrowed)retained_optional.emplace_back(borrowed);
    try{
        if(rebuild_graph){graph.reset();source_discontinuity.clear();rebuild_graph=false;controls.reset=1;}
        if(frame_info_pending){
            const auto discontinuity=source_discontinuity.observe({next_frame_info.stream,next_frame_info.sequence,next_frame_info.timestamp_ns,width,height},controls.reset,false);
            controls.reset=controls.reset||discontinuity!=phoenix_pipeline::ResetReason::None;
        }
        if(pipeline.motion_enabled||pipeline.scene_detection){
            if(!motion_provider)motion_provider=std::make_unique<phoenix_pipeline::D3D12Motion>(device.Get(),queue.Get(),fence.Get());
            const auto estimated=motion_provider->prepare(color,color_state,
                (frame_info_pending?next_frame_info.engine_motion_valid!=0:engine_motion_valid)?motion:nullptr,
                pipeline,controls.reset,frame_info_pending?next_frame_info.timestamp_ns:int64_t(completed_frames)*16666667,
                frame_info_pending?next_frame_info.stream:0,frame_info_pending?std::optional<uint64_t>(next_frame_info.sequence):std::nullopt);
            motion=estimated.motion;motion_state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            controls.reset=controls.reset||estimated.reset;motion_status=motion_provider->status();
        }
        elapsed(1);
        if(!graph)initialize_graph();
        auto& g=*graph;g.executed=Json::array();
        auto* current=g.resample("source",color,color_state,width,height);
        ID3D12Resource* canonical_motion=motion?g.resample("motion",motion,motion_state,width,height,true):nullptr;
        elapsed(0);
        unsigned w=width,h=height;
        ID3D12Resource* neural_source=nullptr;
        for(const auto step:schedule){
            const auto name=std::to_string(unsigned(step.stage));
            g.executed.push_back({{"stage",unsigned(step.stage)},{"width",step.width},{"height",step.height}});
            if(step.stage==phoenix_pipeline::Stage::Neural){
                neural_source=current;
                current=g.neural("neural",current,canonical_motion,controls.reset,controls,w,h,tuning.neural_scale/100);
            }
            else if(step.stage==phoenix_pipeline::Stage::Fusion){
                // All branches see one immutable input and have independent
                // recurrence. Average full-size predictions, never their histories.
                const bool reuse=phoenix_pipeline::fusion_reuses_neural(pipeline,width,height,output_width,output_height);
                auto* base=reuse?neural_source:current;
                auto* mixed=reuse?current:g.neural("fusion0",base,canonical_motion,controls.reset,controls,w,h,tuning.neural_scale/100);
                for(int layer=1;layer<pipeline.fusion_layers;++layer){
                    const float scale=layer==1?pipeline.fusion_second_scale:pipeline.fusion_third_scale;
                    auto* prediction=g.neural("fusion"+std::to_string(layer),base,canonical_motion,controls.reset,controls,w,h,scale);
                    auto* target=g.image("fusion/blend"+std::to_string(layer),w,h);
                    g.run(mixed,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,target,3,prediction,1.f/float(layer+1));mixed=target;
                }
                current=mixed;
            }else if(step.stage==phoenix_pipeline::Stage::Color){
                if(pipeline.color_brightness!=0||pipeline.color_contrast!=1||pipeline.color_saturation!=1||pipeline.color_gamma!=1){
                    auto* treated=g.image("color",w,h);
                    g.run(current,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,treated,2,nullptr,
                        pipeline.color_brightness,pipeline.color_contrast,pipeline.color_saturation,pipeline.color_gamma);current=treated;
                }
            }else{
                auto* scaled=g.image("scale",step.width,step.height);
                if(pipeline.chain_enabled&&pipeline.scaler==3){
                    require(step.width==w*2&&step.height==h*2,"Tiny reconstruction requires 2x output geometry");
                    if(!tiny)tiny=std::make_unique<phoenix_pipeline::TinyUpscaler>(device.Get(),w,h,tiny_model_path);
                    begin();tiny->record(command.Get(),current,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        scaled,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);submit();wait(submitted);scaler_status="Tiny native 2x (experimental)";
                }else if(pipeline.chain_enabled&&pipeline.scaler==1){
                    if(!nis)nis=std::make_unique<phoenix_pipeline::NisScaler>(device.Get());
                    begin();const bool done=nis->record(command.Get(),current,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,scaled,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,pipeline.sharpness);
                    require(done,"NIS admits 1x to 2x output per axis");submit();wait(submitted);scaler_status="NVIDIA Image Scaling";
                }else{g.run(current,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,scaled);scaler_status="Existing spatial scaler";}
                current=scaled;w=step.width;h=step.height;
            }
            elapsed(step.stage==phoenix_pipeline::Stage::Scale?3:2);
        }
        require(w==output_width&&h==output_height,"Processing-chain output geometry differs from the requested target");
        if(pipeline.motion_view&&motion_provider){
            auto* diagnostic=g.image("motion/diagnostic",w,h);
            begin();phoenix_native::transition(command.Get(),diagnostic,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            const bool visualized=motion_provider->visualize(command.Get(),diagnostic,unsigned(pipeline.motion_view));
            phoenix_native::transition(command.Get(),diagnostic,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            submit();wait(submitted);if(visualized)current=diagnostic;
        }
        auto* finished=g.image("finished",w,h);
        // Neural kernels do not reconstruct opacity. Carry original alpha
        // through arbitrary stage order and output-size changes explicitly.
        auto* alpha=g.resample("alpha",color,color_state,w,h);
        g.run(current,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,finished,5,alpha);
        current=finished;
        begin();phoenix_native::transition(command.Get(),current,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
        phoenix_native::transition(command.Get(),output,output_state,D3D12_RESOURCE_STATE_COPY_DEST);
        command->CopyResource(output,current);
        phoenix_native::transition(command.Get(),output,D3D12_RESOURCE_STATE_COPY_DEST,output_state);
        phoenix_native::transition(command.Get(),current,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        submit();wait(submitted);if(motion_provider)motion_provider->consumed(submitted);
        elapsed(4);
        ++completed_frames;frame_info_pending=false;history=motion!=nullptr;retained_optional.clear();
        completed_tuning=tuning.json();completed_pipeline=phoenix_pipeline::options_json(pipeline);
        last_advanced=extended||configuration.contains("tuning");last_intensity=tuning.intensity;last_style=tuning.style;
        call_ms=measured?std::chrono::duration<double,std::milli>(Clock::now()-started).count():0;
        completion_age_ms=monitored?pending_source_age_ms+call_ms:0;
        completed_caller_latency=monitored&&caller_latency_pending;
        if(!g.networks.empty())packed_weights_sha256=g.networks.begin()->second->packed_weights_sha256;
        if(monitored){
            auto budget=pipeline;
            if(tuning.max_time_ms>0)budget.latency_ms=std::min(budget.latency_ms,tuning.max_time_ms);
            if(tuning.target_fps>0)budget.latency_ms=std::min(budget.latency_ms,1000/tuning.target_fps);
            const auto predicted_submit_age=completion_age_ms+previous_presentation_tail_ms;
            const auto knob=latency_policy.sample(budget,predicted_submit_age,configuration.value("offline",false),{
                pipeline.motion_enabled&&pipeline.flow_height>pipeline.minimum_flow_height,
                tuning.neural_scale>pipeline.minimum_scale,tuning.blocks>tuning.minimum_blocks,tuning.passes>unsigned(pipeline.minimum_passes),
                pipeline.flow_height<requested_pipeline.flow_height,tuning.neural_scale<requested_tuning.neural_scale,
                tuning.blocks<requested_tuning.blocks,tuning.passes<requested_tuning.passes});
            using Knob=phoenix_pipeline::BudgetKnob;
            if(knob==Knob::Flow)pipeline.flow_height=std::max(pipeline.minimum_flow_height,pipeline.flow_height/2);
            else if(knob==Knob::Passes)--tuning.passes;
            else if(knob==Knob::Blocks)--tuning.blocks;
            else if(knob==Knob::Scale)tuning.neural_scale=std::max(pipeline.minimum_scale,tuning.neural_scale-10);
            else if(knob==Knob::RestoreFlow)pipeline.flow_height=std::min(requested_pipeline.flow_height,pipeline.flow_height*2);
            else if(knob==Knob::RestorePasses)++tuning.passes;
            else if(knob==Knob::RestoreBlocks)++tuning.blocks;
            else if(knob==Knob::RestoreScale)tuning.neural_scale=std::min(requested_tuning.neural_scale,tuning.neural_scale+10);
            rebuild_graph=knob!=Knob::None&&knob!=Knob::Flow&&knob!=Knob::RestoreFlow;
            pipeline_telemetry={{"stage_wall_ms",stages},{"stage_names",{"capture","motion","neural/color/fusion","scaling","output"}},
                {"processing_call_ms",call_ms},{"budget_ms",budget.latency_ms},{"last_adjustment",int(knob)},
                {"source_age_at_sdk_ms",pending_source_age_ms},{"source_to_sdk_completion_ms",completion_age_ms},
                {"previous_presentation_tail_ms",previous_presentation_tail_ms},{"predicted_present_return_age_ms",predicted_submit_age},
                {"caller_latency_available",completed_caller_latency},
                {"scope","caller eligible-source age plus synchronous processing; prediction uses preceding presentation tail; excludes intentional caller holds, upstream capture/decode and display scanout"}};
        }else pipeline_telemetry={{"monitoring",0}};
        update_neural_budget();
        pending_source_age_ms=0;caller_latency_pending=false;
    }catch(...){poisoned=true;in_flight=true;throw;}
}
