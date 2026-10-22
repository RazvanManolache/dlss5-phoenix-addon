// The portable neural engines share the native pipeline's stage graph. Only
// neural evaluation crosses this host adapter; motion, scale, color, fusion,
// HDR conversion and latency policy keep their existing owning stage/session.
void PhoenixNativeSession::update_neural_budget(){
    // The pipeline governor is authoritative when enabled. Otherwise the
    // established FPS/time limits adjust only block count, without enabling
    // optional telemetry or GPU queries. Both paths use one settling policy.
    if(!neural_budget_enabled()||(pipeline.monitoring&&pipeline.priority))return;
    auto budget=pipeline;budget.priority=1;budget.monitoring=1;
    budget.adjust_flow=budget.adjust_scale=budget.adjust_passes=false;budget.adjust_blocks=true;
    budget.latency_ms=tuning.max_time_ms>0?tuning.max_time_ms:1000.f;
    if(tuning.target_fps>0)budget.latency_ms=std::min(budget.latency_ms,1000/tuning.target_fps);
    const bool backend_clock=configuration.value("neural_budget_clock",std::string{})=="backend_gpu";
    if(backend_clock&&!gpu_sampled_last_frame){pipeline_telemetry["neural_budget"]={{"enabled",true},{"sample_available",false},{"clock","backend_gpu"}};return;}
    const double sample=backend_clock?gpu_ms:call_ms;
    const auto decision=latency_policy.sample(budget,sample,false,{false,false,
        tuning.blocks>tuning.minimum_blocks,false,false,false,tuning.blocks<requested_tuning.blocks,false});
    using Knob=phoenix_pipeline::BudgetKnob;
    if(decision==Knob::Blocks)--tuning.blocks;
    else if(decision==Knob::RestoreBlocks)++tuning.blocks;
    if(decision==Knob::Blocks||decision==Knob::RestoreBlocks){
        transformer_blocks=tuning.blocks;
        if(graph_owner)rebuild_graph=true;
        else {history=false;if(network)network->invalidate();}
    }
    pipeline_telemetry["neural_budget"]={{"enabled",true},{"budget_ms",budget.latency_ms},
        {"processing_ms",sample},{"next_blocks",tuning.blocks},{"last_adjustment",int(decision)},
        {"clock",backend_clock?"backend_gpu":"source_call"},
        {"scope",backend_clock?"portable backend GPU network samples; excludes host transfers and presentation":
            "source processing call; excludes caller capture/presentation and host API outer transfers"}};
}
Json PhoenixNativeSession::portable_stage_reports()const{
    Json reports=Json::array();
    auto report_for=[](const PhoenixNativeSession& stage){
        auto report=stage.portable_status;
        report["inference_pid"]=stage.portable_session?stage.portable_session->process_id():0;
        report["worker_execution"]="native_broker";
        return report;
    };
    if(portable_leaf)reports.push_back(report_for(*this));
    if(graph)for(const auto& [name,stage]:graph->networks)if(stage->portable_leaf){
        auto report=report_for(*stage);report["stage"]=name;reports.push_back(std::move(report));
    }
    return reports;
}
void PhoenixNativeSession::frame_portable(ID3D12Resource* color,D3D12_RESOURCE_STATES color_state,
    ID3D12Resource* motion,D3D12_RESOURCE_STATES motion_state,ID3D12Resource* output,
    D3D12_RESOURCE_STATES output_state,const PhoenixNativeControls* supplied,const PhoenixNativeFrameV2* extended){
    validate(color,DXGI_FORMAT_R16G16B16A16_FLOAT);validate(output,DXGI_FORMAT_R16G16B16A16_FLOAT,true);
    if(motion)validate(motion,DXGI_FORMAT_R16G16_FLOAT);
    require(color!=output&&motion!=output&&motion!=color,"Portable neural texture alias rejected");
    require(!extended||(!extended->control_mask.resource&&!extended->ui.resource&&!extended->ui_alpha.resource&&
        !extended->backbuffer.resource&&!extended->distortion.resource),"Portable neural stage accepts canonical color/motion only");
    PhoenixNativeControls controls;phoenix_native_default_controls(&controls);if(supplied)controls=*supplied;
    require(controls.struct_bytes==sizeof(controls),"Portable neural control structure mismatch");
    Json parameters{{"local_tone",controls.local_tone},{"local_structure",controls.local_structure},
        {"skin_structure",controls.skin_structure},{"auto_mask",bool(controls.auto_mask)},
        {"intensity",extended?extended->intensity:1.f},{"style",extended?extended->style:0u}};
    phoenix_cpp::Controls::from(parameters); // Reject before touching recurrence.
    const auto counts=configuration.value("repeats",std::array<int,8>{1,1,1,1,1,1,1,1});
    const size_t pixels=size_t(width)*height;
    auto prepare=[&](Plane& plane,ID3D12Resource* image,bool readback){
        if(plane.transfer)return;
        const auto description=image->GetDesc();
        device->GetCopyableFootprints(&description,0,1,0,&plane.footprint,nullptr,nullptr,&plane.bytes);
        plane.transfer=phoenix_native::buffer(device.Get(),plane.bytes,
            readback?D3D12_HEAP_TYPE_READBACK:D3D12_HEAP_TYPE_UPLOAD,
            readback?D3D12_RESOURCE_STATE_COPY_DEST:D3D12_RESOURCE_STATE_GENERIC_READ);
        phoenix_native::memory::resource(device.Get(),plane.transfer.Get(),accounting,phoenix_native::memory::Kind::upload);
    };
    prepare(portable_color_readback,color,true);if(motion)prepare(portable_motion_readback,motion,true);
    prepare(portable_output_upload,output,false);
    const auto started=pipeline.monitoring?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
    std::vector<float> rgb(pixels*3),vectors(motion?pixels*2:0),result(pixels*3);
    std::vector<uint16_t> rgba(pixels*4);
    phx::Kernel<HostMath> math{};
    try{
        begin();
        const auto download=[&](Plane& plane,ID3D12Resource* image,D3D12_RESOURCE_STATES state){
            phoenix_native::transition(command.Get(),image,state,D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION from{image,D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0},
                to{plane.transfer.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,plane.footprint};
            command->CopyTextureRegion(&to,0,0,0,&from,nullptr);
            phoenix_native::transition(command.Get(),image,D3D12_RESOURCE_STATE_COPY_SOURCE,state);
        };
        download(portable_color_readback,color,color_state);
        if(motion)download(portable_motion_readback,motion,motion_state);
        submit();wait(submitted);
        auto read=[&](Plane& plane,unsigned channels,auto consume){
            void* mapped=nullptr;D3D12_RANGE range{0,size_t(plane.bytes)},empty{};
            hr(plane.transfer->Map(0,&range,&mapped));
            for(unsigned y=0;y<height;++y){
                const auto* row=reinterpret_cast<const uint16_t*>(static_cast<const char*>(mapped)+plane.footprint.Offset+size_t(y)*plane.footprint.Footprint.RowPitch);
                for(unsigned x=0;x<width;++x)for(unsigned c=0;c<channels;++c)consume((size_t(y)*width+x)*channels+c,row[x*channels+c]);
            }
            plane.transfer->Unmap(0,&empty);
        };
        read(portable_color_readback,4,[&](size_t i,uint16_t value){rgba[i]=value;if(i%4<3)rgb[i/4*3+i%4]=std::clamp(math.half_value(value),0.f,1.f);});
        if(motion)read(portable_motion_readback,2,[&](size_t i,uint16_t value){vectors[i]=math.half_value(value)*(i%2?controls.motion_scale_y:controls.motion_scale_x);});
        Json metadata{{"reset",bool(controls.reset)||!history||!motion},{"has_motion",bool(motion)},
            {"has_depth",false},{"has_history_validity",false},{"controls",parameters},
            {"transformer_blocks",transformer_blocks},{"repeats",counts},{"passes",1},
            {"adaptive_budget",{{"enabled",false}}}};
        require(portable_session->frame(rgb.data(),vectors.empty()?nullptr:vectors.data(),nullptr,nullptr,metadata.dump(),result.data())==0,
            portable_session->error());
        portable_status=Json::parse(portable_session->status_json());
        require(std::all_of(result.begin(),result.end(),[](float value){return std::isfinite(value);}),"Portable runner returned nonfinite pixels");
        for(size_t p=0;p<pixels;++p)for(unsigned c=0;c<3;++c)rgba[p*4+c]=uint16_t(math.half_bits(result[p*3+c]));
        void* mapped=nullptr;D3D12_RANGE empty{};hr(portable_output_upload.transfer->Map(0,&empty,&mapped));
        for(unsigned y=0;y<height;++y)std::memcpy(static_cast<char*>(mapped)+portable_output_upload.footprint.Offset+
            size_t(y)*portable_output_upload.footprint.Footprint.RowPitch,rgba.data()+size_t(y)*width*4,size_t(width)*8);
        portable_output_upload.transfer->Unmap(0,nullptr);
        begin();phoenix_native::transition(command.Get(),output,output_state,D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION from{portable_output_upload.transfer.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,portable_output_upload.footprint},
            to{output,D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0};
        command->CopyTextureRegion(&to,0,0,0,&from,nullptr);
        phoenix_native::transition(command.Get(),output,D3D12_RESOURCE_STATE_COPY_DEST,output_state);
        submit();wait(submitted);
        history=motion!=nullptr;frame_info_pending=false;++completed_frames;
        const auto timing=portable_status.value("network_schedule",Json::object());
        const bool budget_clock=configuration.value("neural_budget_clock",std::string{})=="backend_gpu";
        gpu_sampled_last_frame=(measure_gpu||budget_clock)&&timing.value("gpu_sample",false);gpu_ms=gpu_sampled_last_frame?timing.value("gpu_ms",0.):0;
        call_ms=pipeline.monitoring?std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count():0;
    }catch(...){poisoned=true;throw;}
}
