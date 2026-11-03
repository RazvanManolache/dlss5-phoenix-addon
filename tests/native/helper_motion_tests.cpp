// WARP only. Exercise engine/zero/scene services without opening NVIDIA OFA or a model.
#include "d3d12_test_context.h"
#include "../../src/integration/compatibility/host/phoenix-helper-motion.h"
#include <iostream>

int main() try {
    unsigned checks=0;
    auto check=[&](bool ok){++checks;if(!ok)throw std::runtime_error("Helper motion check "+std::to_string(checks));};
    phoenix_test::Gpu gpu; // Software adapter; never select the physical GPU.
    auto color=gpu.texture(32,24),engine=gpu.texture(32,24,true);
    auto original=gpu.readback(color.Get());
    phoenix_native::transition(gpu.cmd.Get(),color.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON);
    phoenix_native::transition(gpu.cmd.Get(),engine.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON);
    gpu.submit();
    PhoenixHelperMotion helper;phoenix_pipeline::Options options;
    options.motion_enabled=false; // This software fixture must not initialize OFA.
    options.scene_detection=true; // Small scene readback only, no optical-flow driver.
    helper.configure(options);
    auto frame=[&](unsigned stream,uint64_t capture,FeedGuideState guides={}){
        return helper.prepare(stream,capture,color.Get(),engine.Get(),false,gpu.device.Get(),gpu.queue.Get(),gpu.fence.Get(),guides);
    };
    auto zero=frame(0,1);check(zero.reset&&!zero.estimated&&zero.motion!=engine.Get());
    check(!helper.snapshot().at("history_available").get<bool>());
    auto first=frame(0,2,{1,1,1});check(first.reset&&!first.estimated&&first.motion==engine.Get());
    auto next=frame(0,3,{1,1,1});check(!next.reset&&!next.estimated&&next.motion==engine.Get());
    check(helper.snapshot().at("history_available").get<bool>());
    auto repeat=frame(0,3,{1,1,1});check(repeat.reset&&repeat.motion!=engine.Get());
    auto missing=frame(0,4);check(missing.reset&&missing.motion==zero.motion);
    auto eye=frame(1,1,{1,1,1});check(eye.reset&&eye.motion==engine.Get());
    check(!frame(1,2,{1,1,1}).reset);
    helper.request_reset();check(frame(1,3,{1,1,1}).reset);
    options.motion_source=int(phoenix_pipeline::MotionSource::Zero);helper.configure(options);
    check(frame(0,5,{1,1,1}).motion==zero.motion); // Explicit zero ignores real guides too.
    check(!frame(0,6,{1,1,1}).estimated);
    auto status=helper.snapshot();check(status.at("streams").at(0).at("estimated_frames")==0);
    check(status.at("streams").at(0).at("analysis_width")==32);
    // Source stays unchanged; the private fallback is actually all-zero FP16.
    gpu.next();
    phoenix_native::transition(gpu.cmd.Get(),color.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    phoenix_native::transition(gpu.cmd.Get(),zero.motion,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto unchanged=gpu.readback(color.Get()),zeros=gpu.readback(zero.motion);gpu.submit();
    auto bytes=[](ID3D12Resource* resource){
        std::vector<uint8_t> result(size_t(resource->GetDesc().Width));void* mapped=nullptr;
        D3D12_RANGE range{0,result.size()};phoenix_native::hr(resource->Map(0,&range,&mapped));
        std::memcpy(result.data(),mapped,result.size());D3D12_RANGE empty{};resource->Unmap(0,&empty);return result;
    };
    check(bytes(original.first.Get())==bytes(unchanged.first.Get()));
    const auto zero_bytes=bytes(zeros.first.Get());check(std::all_of(zero_bytes.begin(),zero_bytes.end(),[](auto v){return v==0;}));
    gpu.validate();helper.clear();check(helper.snapshot().at("streams").empty());
    std::cout<<nlohmann::json{{"passed",true},{"checks",checks},{"adapter","WARP"},{"hardware_inference",false}}.dump()<<'\n';
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
