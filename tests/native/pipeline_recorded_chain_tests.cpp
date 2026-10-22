#include "../../src/pipeline/recorded_chain.h"
#include "../../src/native/support/production_direct.h"
#include "../../src/native/support/comparison.h"
#include <dxgi1_6.h>
#include <d3d12sdklayers.h>
#include <DirectXPackedVector.h>
#include <iostream>
using Microsoft::WRL::ComPtr;
using phoenix_native::hr;
using phoenix_bundle::require;
using Json=nlohmann::json;
using namespace phoenix_pipeline;
#include "d3d12_test_context.h"
using phoenix_test::Gpu;
struct AffineStage final:RecordedChain::NeuralStage {
    phoenix_addon::phoenix::direct::Pass pass;
    ComPtr<ID3D12Resource> output;unsigned width,height;
    AffineStage(ID3D12Device* device,unsigned w,unsigned h):pass(device,image_ops_shader),
        output(phoenix_native::image(device,w,h,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,true)),width(w),height(h){}
    ID3D12Resource* record(ID3D12GraphicsCommandList* cmd,ID3D12Resource* color,ID3D12Resource*,phoenix_addon::Controls,unsigned,ID3D12CommandQueue*)override{
        phoenix_native::transition(cmd,output.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        pass.record(cmd,{color,nullptr,nullptr},output.Get(),ImageConstants{width,height,2,0,.15f,.5f,1,1,0});
        phoenix_native::transition(cmd,output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);return output.Get();
    }
    void commit(ID3D12GraphicsCommandList*)override{}void invalidate()override{}uint64_t launches()const override{return 0;}
};
void fusion_input_oracle(){
    Gpu gpu;auto source=gpu.texture(41,23),motion=gpu.texture(41,23,true);
    for(bool main_stage:{true,false}){
        Options p;p.chain_enabled=true;p.fusion=true;p.enabled={main_stage,false,false,true};p.fusion_second_scale=1;
        RecordedChain chain(gpu.device.Get(),41,23,41,23,p,1,[&](unsigned w,unsigned h){return std::make_unique<AffineStage>(gpu.device.Get(),w,h);});
        phoenix_addon::Controls controls;controls.reset=1;auto* result=chain.record(gpu.cmd.Get(),source.Get(),motion.Get(),controls,8,gpu.queue.Get(),0);
        auto [readback,fp]=gpu.readback(result);gpu.submit();chain.commit(gpu.cmd.Get());
        void* data=nullptr;D3D12_RANGE range{0,size_t(readback->GetDesc().Width)},empty{};hr(readback->Map(0,&range,&data));
        for(unsigned y=0;y<23;++y)for(unsigned x=0;x<41;++x)for(unsigned c=0;c<3;++c){
            const float input=DirectX::PackedVector::XMConvertHalfToFloat(DirectX::PackedVector::XMConvertFloatToHalf(float((x*37+y*11+c*51)%256)/255));
            const auto* pixel=reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(data)+fp.Footprint.RowPitch*y)+x*4;
            require(std::abs(DirectX::PackedVector::XMConvertHalfToFloat(pixel[c])-(input*.5f+.4f))<.002f,
                main_stage?"Fusion branches reprocessed the main prediction":"Standalone fusion omitted its full-resolution prediction");
        }
        readback->Unmap(0,&empty);gpu.validate();gpu.next();
    }
}
Json native_case(const std::filesystem::path& folder,const std::filesystem::path& dll,const std::string& model,int eighth_repeats=1){
    Gpu gpu(true);auto color=gpu.texture(160,96),motion=gpu.texture(160,96,true);
    auto source=phoenix_model::WeightSource::config({{"kind","nvidia_dll"},{"path",dll.string()},{"model",model},{"sha256",phoenix_model::mapped_file_digest(dll)}});
    phoenix_bundle::Bundle bundle(folder,phoenix_model::mapped_file_digest,source);
    const auto kernels=phoenix_kernel::load_catalog(phoenix_native::catalog(bundle));const auto weights=bundle.load_weights();
    auto select=[&](unsigned w,unsigned h){return Json(bundle.select_frame(w,h,[&](const Json& frame){phoenix_native::preflight(frame,kernels);}));};
    phoenix_addon::phoenix::Settings settings;settings.processing.enabled=true;settings.processing.encoding=2;settings.processing.scale=50;
    settings.pipeline.chain_enabled=true;settings.pipeline.fusion=true;settings.pipeline.enabled[3]=true;settings.pipeline.order={1,0,3,2};settings.pipeline.color_brightness=.03f;
    settings.block_repeats[7]=eighth_repeats;
    phoenix_addon::phoenix::direct::Runner runner(gpu.device.Get(),gpu.queue.Get(),select(80,48),weights,kernels,
        DXGI_FORMAT_R16G16B16A16_FLOAT,DXGI_FORMAT_R16G16_FLOAT,false,1,settings.block_repeats,{},&settings,{},true,160,96,{},false,select);
    phoenix_addon::FrameEvidence evidence;evidence.queue=gpu.queue.Get();evidence.color_state=evidence.motion_state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    evidence.states_verified=true;evidence.experimental_appearance_contract=true;evidence.restore_bindings=[](void*,ID3D12GraphicsCommandList*){};
    std::pair<ComPtr<ID3D12Resource>,D3D12_PLACED_SUBRESOURCE_FOOTPRINT> captured;
    for(unsigned frame=0;frame<3;++frame){
        phoenix_addon::Controls controls;controls.reset=frame!=1;controls.style=1;controls.intensity=.65f;
        runner.record(gpu.cmd.Get(),color.Get(),motion.Get(),evidence,controls,settings,8);
        if(frame==2)captured=gpu.readback(color.Get());
        ID3D12CommandList* list=gpu.cmd.Get();runner.validate_submit(gpu.queue.Get(),1,&list);
        gpu.submit(false);require(runner.after_submit(gpu.queue.Get(),1,&list),"Missing addon submission commit");
        hr(gpu.queue->Signal(gpu.fence.Get(),++gpu.serial));gpu.finish();if(frame<2)gpu.next();
    }
    gpu.validate();const auto report=runner.execution_status().at("processing_chain");
    require(report.at("independent_temporal_stages")==2,"Addon fusion does not own two histories");
    require(runner.completed()==3,"Addon completion fence did not retire");
    const auto [readback,fp]=captured;void* mapped=nullptr;D3D12_RANGE range{0,size_t(readback->GetDesc().Width)},empty{};
    hr(readback->Map(0,&range,&mapped));double difference=0;unsigned changed=0;
    std::vector<uint8_t> pixels(160*96*8);
    for(unsigned y=0;y<96;++y)std::memcpy(pixels.data()+y*160*8,static_cast<const uint8_t*>(mapped)+fp.Footprint.RowPitch*y,160*8);
    for(unsigned y=0;y<96;++y)for(unsigned x=0;x<160;++x){auto* pixel=reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(mapped)+fp.Footprint.RowPitch*y)+x*4;
        for(unsigned c=0;c<3;++c){const float actual=DirectX::PackedVector::XMConvertHalfToFloat(pixel[c]);require(std::isfinite(actual),"Nonfinite addon output");
            const float original=DirectX::PackedVector::XMConvertHalfToFloat(DirectX::PackedVector::XMConvertFloatToHalf(float((x*37+y*11+c*51)%256)/255));
            const float delta=std::abs(actual-original);difference+=delta;changed+=delta>.002f;}
        require(pixel[3]==0x3c00,"Addon did not preserve alpha");
    }
    readback->Unmap(0,&empty);require(changed>100,"Addon chain did not change pixels");
    return {{"model",model},{"frames",3},{"independent_stages",2},{"host_list_submissions",true},{"full_output_extent",{160,96}},
        {"scale_before_neural",true},{"fusion",true},{"appearance_controls",true},{"alpha_exact",true},{"mean_rgb_change",difference/(160*96*3)},
        {"eighth_block_repeats",eighth_repeats},{"rgba16f_sha256",phoenix_model::weight_digest(pixels.data(),pixels.size())},{"debug_errors",0}};
}
struct Counts {unsigned records=0,commits=0,resets=0;};
struct TestStage:RecordedChain::NeuralStage {
    Counts& counts;
    explicit TestStage(Counts& c):counts(c){}
    ID3D12Resource* record(ID3D12GraphicsCommandList*,ID3D12Resource* source,ID3D12Resource*,phoenix_addon::Controls c,unsigned,ID3D12CommandQueue*)override{
        ++counts.records;if(c.reset)++counts.resets;return source;
    }
    void commit(ID3D12GraphicsCommandList*)override{++counts.commits;}
    void invalidate()override{}
    uint64_t launches()const override{return 0;}
};
int main(int argc,char** argv){try{
    fusion_input_oracle();
    if(argc==3){Json cases=Json::array();for(const auto model:{"original","small-fp8"}){
        auto baseline=native_case(std::filesystem::absolute(argv[1]),std::filesystem::absolute(argv[2]),model);
        for(int repeats:{0,2}){auto changed=native_case(std::filesystem::absolute(argv[1]),std::filesystem::absolute(argv[2]),model,repeats);
            require(changed.at("rgba16f_sha256")!=baseline.at("rgba16f_sha256"),"Eighth central block control did not affect addon pixels");cases.push_back(changed);}
        cases.push_back(baseline);}
        std::cout<<Json{{"passed",true},{"hardware",true},{"cases",cases}}.dump()<<'\n';return 0;}
    Gpu gpu;auto source=gpu.texture(41,23),motion=gpu.texture(41,23,true);
    Options p;p.chain_enabled=true;p.enabled={false,true,true,false};p.color_brightness=.15f;p.color_contrast=1.7f;p.color_saturation=.6f;p.color_gamma=.8f;
    p.order={2,1,0,3};RecordedChain first(gpu.device.Get(),41,23,82,46,p,1,{});
    p.order={1,2,0,3};RecordedChain second(gpu.device.Get(),41,23,82,46,p,1,{});
    phoenix_addon::Controls controls;controls.reset=1;
    auto* a=first.record(gpu.cmd.Get(),source.Get(),motion.Get(),controls,8,gpu.queue.Get(),0);
    auto* b=second.record(gpu.cmd.Get(),source.Get(),motion.Get(),controls,8,gpu.queue.Get(),0);
    auto [ar,af]=gpu.readback(a);auto [br,bf]=gpu.readback(b);
    phoenix_addon::phoenix::Comparison zoom_view(gpu.device.Get(),41,23,DXGI_FORMAT_R16G16B16A16_FLOAT),loupe_view(gpu.device.Get(),41,23,DXGI_FORMAT_R16G16B16A16_FLOAT);
    auto zoom_output=gpu.texture(41,23),loupe_output=gpu.texture(41,23);
    Options zoom;zoom.compare_zoom=4;zoom.loupe_x=.25f;zoom.loupe_y=.75f;
    const std::array<ID3D12Resource*,3> views{source.Get(),source.Get(),source.Get()};
    const std::array<D3D12_RESOURCE_STATES,3> states{D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
    zoom_view.record(gpu.cmd.Get(),views,states,zoom_output.Get(),states[0],1,0,2,.5f,1,zoom);
    auto [zr,zf]=gpu.readback(zoom_output.Get());zoom.loupe=true;
    loupe_view.record(gpu.cmd.Get(),views,states,loupe_output.Get(),states[0],1,0,2,.5f,1,zoom);
    auto [lr,lf]=gpu.readback(loupe_output.Get());
    std::vector<std::unique_ptr<Counts>> counts;
    Options fusion;fusion.chain_enabled=true;fusion.fusion=true;fusion.enabled={false,false,false,true};fusion.fusion_layers=3;
    RecordedChain fused(gpu.device.Get(),41,23,41,23,fusion,2,[&](unsigned,unsigned){counts.push_back(std::make_unique<Counts>());return std::make_unique<TestStage>(*counts.back());});
    for(unsigned frame=0;frame<3;++frame){
        controls.reset=frame!=1;fused.record(gpu.cmd.Get(),source.Get(),motion.Get(),controls,8,gpu.queue.Get(),frame);
        gpu.submit(frame==2);fused.commit(gpu.cmd.Get());if(frame<2)gpu.next();
    }
    require(counts.size()==6,"Passes/branches do not own separate instances");
    for(const auto& c:counts)require(c->records==3&&c->commits==3&&c->resets==2,"Reset/submission fanout failed");
    gpu.validate();
    auto verify_view=[&](ID3D12Resource* readback,const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp,bool loupe){
        void* data=nullptr;D3D12_RANGE range{0,size_t(readback->GetDesc().Width)},empty{};hr(readback->Map(0,&range,&data));
        const float cx=41.f*32/127,cy=23.f*95/127;unsigned moved=0;
        for(unsigned y=0;y<23;++y)for(unsigned x=0;x<41;++x){
            unsigned sx=x,sy=y;const float dx=x+.5f-cx,dy=y+.5f-cy;
            if(!loupe||dx*dx+dy*dy<=23.f*.18f*23.f*.18f){
                sx=unsigned(std::clamp(std::floor((x+.5f-cx)/4+cx-.5f+.5f),0.f,40.f));
                sy=unsigned(std::clamp(std::floor((y+.5f-cy)/4+cy-.5f+.5f),0.f,22.f));
            }
            moved+=sx!=x||sy!=y;
            auto* pixel=reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(data)+fp.Footprint.RowPitch*y)+x*4;
            for(unsigned c=0;c<3;++c){const auto expected=DirectX::PackedVector::XMConvertFloatToHalf(float((sx*37+sy*11+c*51)%256)/255);require(pixel[c]==expected,"Comparison focus/zoom pixel oracle failed");}
        }
        readback->Unmap(0,&empty);require(moved>10,"Comparison view did not magnify pixels");
    };
    verify_view(zr.Get(),zf,false);verify_view(lr.Get(),lf,true);
    void *av=nullptr,*bv=nullptr;D3D12_RANGE range{0,size_t(ar->GetDesc().Width)},empty{};
    hr(ar->Map(0,&range,&av));hr(br->Map(0,&range,&bv));float maximum=0;
    for(unsigned y=0;y<46;++y)for(unsigned x=0;x<82*4;++x){
        const auto* ap=reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(av)+af.Footprint.RowPitch*y);
        const auto* bp=reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(bv)+bf.Footprint.RowPitch*y);
        maximum=std::max(maximum,std::abs(DirectX::PackedVector::XMConvertHalfToFloat(ap[x])-DirectX::PackedVector::XMConvertHalfToFloat(bp[x])));
    }
    ar->Unmap(0,&empty);br->Unmap(0,&empty);require(maximum>.01,"Stage order failed to change pixels");
    std::cout<<Json{{"passed",true},{"warp",true},{"recorded_on_host_list",true},{"stage_order_max_difference",maximum},
        {"independent_stages",6},{"fusion_same_source_oracle",true},{"standalone_fusion_full_prediction",true},{"reset_fanout",true},{"three_queued_frames",true},{"comparison_zoom_loupe_exact",true},{"debug_errors",0}}.dump()<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
