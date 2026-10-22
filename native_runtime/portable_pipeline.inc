// Included after Runtime. Host stages surround the selected compiled backend;
// no Python, ONNX runtime or fixed-resolution model is used.
void Runtime::run_pipeline(const float* color,const float* motion,const float* depth,const float* validity,
    const Json& metadata,float* output,const float* mask,uint64_t adapter_bytes,const AdvancedFrame* advanced){
    using namespace phoenix_pipeline;
    const auto options=without_comparison(options_from_json(metadata.at("pipeline")));
    demand(options.priority==0,"Portable custom chains currently require fixed pipeline quality");
    demand(!options.chain_enabled||!options.enabled[unsigned(Stage::Scale)]||options.scaler==0,
        "Portable host chain requires the existing spatial scaler");
    demand(!options.motion_view,"Portable host chain does not provide motion diagnostic images");
    demand(!advanced||(!advanced->ui&&!advanced->ui_alpha&&!advanced->backbuffer&&!advanced->distortion),
        "Portable custom chains require UI/distortion composition outside this frame API");
    demand(color&&output,"Missing frame buffers");
    const unsigned width=frame->width,height=frame->height;
    const size_t count=size_t(width)*height;
    const bool has_motion=metadata.value("has_motion",false),has_depth=metadata.value("has_depth",false),
        has_validity=metadata.value("has_history_validity",false);
    demand(!has_motion||motion,"Missing declared motion");
    demand(!has_depth||depth,"Missing declared depth");
    demand(!has_validity||(has_motion&&validity),"History validity requires motion and a validity plane");
    demand(!metadata.value("has_control_mask",false)||mask,"Missing declared control mask");
    const auto output_stage=metadata.value("output_stage",std::string("presented"));
    demand(output_stage=="presented"||output_stage=="neural_only","Unknown output stage");
    Controls::from(metadata.value("controls",Json::object()));
    auto next_schedule=schedule;
    const auto device=status.at("device").get<std::string>();
    next_schedule.prepare(metadata,status.at("model_resource_sha256").get<std::string>()+":host-chain",
        device!="cpu"&&device!="sycl:cpu");
    for(size_t p=0;p<count;++p){
        for(unsigned c=0;c<3;++c)demand(std::isfinite(color[p*3+c])&&color[p*3+c]>=0&&color[p*3+c]<=1,"Invalid RGB proxy input");
        if(has_motion)demand(std::isfinite(motion[p*2])&&std::isfinite(motion[p*2+1]),"Nonfinite motion");
        if(has_depth)demand(std::isfinite(depth[p]),"Nonfinite depth");
        if(has_validity)demand(validity[p]==0||validity[p]==1,"Nonbinary validity");
    }
    validate_control_mask(mask,count);
    const bool explicit_reset=metadata.value("reset",false);
    demand(!pipeline_failed||explicit_reset,"Portable chain failed; reset or recreate the runtime before retrying");
    // Every rejection above precedes changes to any stage or temporal history.
    const auto signature=options_json(options);
    if(!pipeline_active)pipeline_base_status=status;
    const bool reset=explicit_reset||pipeline_signature!=signature||!pipeline_active||pipeline_failed;
    if(reset){pipeline_networks.clear();pipeline_failed=false;}
    pipeline_signature=signature;pipeline_active=true;
    auto request=metadata;request.erase("pipeline");request["reset"]=reset;
    request["adaptive_budget"]={{"enabled",false}};
    request["transformer_blocks"]=next_schedule.depth;request["repeats"]=next_schedule.repeats;
    // One parent schedule controls all branches; children cannot adapt against
    // different costs or count generated presentation frames as source frames.
    std::vector<Event> pipeline_events;
    Json executed=Json::array(),stages=Json::array();
    const auto started=Clock::now();
    double prepare_ms=0,execute_ms=0,resolve_ms=0;
    uint64_t resident_bytes=graph?graph->memory_bytes():0,weight_bytes=graph?graph->weight_bytes():0;
    auto neural=[&](const std::string& name,const HostImage& input){
        auto& child=pipeline_networks[name];
        if(!child)child=std::make_unique<Runtime>(configuration,input.width,input.height,
            configuration.at("backend")=="vulkan"?this:nullptr);
        HostImage prediction(input.width,input.height,3);
        auto plane=[&](const float* source,unsigned channels,HostSampling sampling){
            return source?host_resample(HostImage(width,height,channels,source),input.width,input.height,sampling):HostImage{};
        };
        auto vectors=plane(has_motion?motion:nullptr,2,HostSampling::Motion);
        auto z=plane(has_depth?depth:nullptr,1,HostSampling::Nearest);
        auto valid=plane(has_validity?validity:nullptr,1,HostSampling::Validity);
        auto controls=plane(mask,4,HostSampling::Image);
        child->run(input.pixels.data(),vectors.pixels.empty()?nullptr:vectors.pixels.data(),
            z.pixels.empty()?nullptr:z.pixels.data(),valid.pixels.empty()?nullptr:valid.pixels.data(),request,
            prediction.pixels.data(),controls.pixels.empty()?nullptr:controls.pixels.data(),adapter_bytes,nullptr);
        const auto& report=child->status;
        prepare_ms+=report.value("prepare_ms",0.);execute_ms+=report.value("execute_with_transfers_ms",0.);
        resolve_ms+=report.value("resolve_ms",0.);
        resident_bytes+=report.value("resident_bytes",uint64_t(0));weight_bytes+=report.value("resident_weight_bytes",uint64_t(0));
        const auto timing=report.at("network_schedule");
        if(timing.value("gpu_sample",false))pipeline_events.push_back({0,0,timing.at("gpu_ms").get<double>(),true,0});
        stages.push_back({{"name",name},{"width",input.width},{"height",input.height},
            {"passes",report.at("passes")},{"history_used",report.at("history_used")}});
        return prediction;
    };
    try{
        auto result=host_chain(options,HostImage(width,height,3,color),width,height,neural,&executed);
        demand(std::all_of(result.pixels.begin(),result.pixels.end(),[](float v){return std::isfinite(v);}),"Nonfinite portable chain result");
        next_schedule.complete(pipeline_events);
        std::copy(result.pixels.begin(),result.pixels.end(),output);
        std::lock_guard guard(lock);
        if(!pipeline_networks.empty()){
            const auto selected=pipeline_networks.contains("neural")?"neural":pipeline_networks.begin()->first;
            status=pipeline_networks.at(selected)->status;
        }else{
            status=pipeline_base_status;
            status.update({{"history_used",false},{"inference_validated",false},{"passes",0},
                {"command_count",0},{"operator_ms",0.},{"layers",Json::array()}});
        }
        schedule=std::move(next_schedule);
        status.update({{"state","completed"},{"pipeline",signature},{"executed_chain",executed},
            {"chain_execution","native C++ host stages around selected portable backend"},
            {"independent_temporal_stages",pipeline_networks.size()*schedule.passes},
            {"pipeline_stages",stages},{"pipeline_histories_reset",reset},
            {"network_schedule",schedule.status()},{"worker_ms",elapsed(started)},
            {"prepare_ms",prepare_ms},{"execute_with_transfers_ms",execute_ms},{"resolve_ms",resolve_ms},
            {"resident_bytes",resident_bytes},{"resident_weight_bytes",weight_bytes},
            {"resident_scratch_bytes",resident_bytes-weight_bytes},
            {"pipeline_host_stage_scope","CPU resampling/color/fusion; backend inference and transfers measured separately; fixed frame ABI extent"}});
    }catch(...){pipeline_failed=true;throw;}
}
