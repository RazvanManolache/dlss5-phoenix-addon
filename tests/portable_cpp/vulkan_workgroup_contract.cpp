// Compare the optional mobile Vulkan route with CPU at every model checkpoint.
#include "../../native_runtime/graph.h"
#include "../../native_runtime/frame.h"
#include "../../native_runtime/platform.h"
#include "../../native_runtime/backend_options.h"
#include <iostream>
#include <cstring>
using namespace phoenix_cpp;

int main(int argc, char** argv) {
    try {
        demand(argc == 2, "Usage: vulkan_workgroup_contract JOB.json");
        Json job; std::ifstream(argv[1]) >> job;
        auto cfg = job.at("config");
        auto root = std::filesystem::path(cfg.at("runtime_root").get<std::string>());
        auto cpu_module = open_library(root / library_name("phoenix_cpu_cpp"));
        auto gpu_module = open_library(root / library_name("phoenix_vulkan_cpp"));
        demand(cpu_module && gpu_module, "Backend libraries unavailable");
        auto cpu_create = reinterpret_cast<Backend*(*)(const char*,unsigned,char*,size_t)>(
            library_symbol(cpu_module,"phoenix_backend_create_threads"));
        auto gpu_create = reinterpret_cast<Backend*(*)(const char*,const char*,uint32_t,char*,size_t)>(
            library_symbol(gpu_module,"phoenix_backend_create_utf8"));
        auto cpu_destroy = reinterpret_cast<DestroyBackend>(library_symbol(cpu_module,"phoenix_backend_destroy"));
        auto gpu_destroy = reinterpret_cast<DestroyBackend>(library_symbol(gpu_module,"phoenix_backend_destroy"));
        demand(cpu_create && gpu_create && cpu_destroy && gpu_destroy,"Backend factories unavailable");
        char error[2048]{};
        std::unique_ptr<Backend,DestroyBackend> cpu(cpu_create("cpu",4,error,sizeof(error)),cpu_destroy);
        demand(bool(cpu),error);
        auto folder=(root/"vulkan").string(),device=cfg.at("device").get<std::string>();
        uint32_t options=VulkanWorkgroupCompatibility|VulkanTiledFallback|VulkanCoalescedFallback;
        std::unique_ptr<Backend,DestroyBackend> gpu(gpu_create(device.c_str(),folder.c_str(),options,error,sizeof(error)),gpu_destroy);
        demand(bool(gpu),error);
        unsigned width=job.at("width"),height=job.at("height");
        // This contract accepts the museum's 320x144 fixture, whose runtime uses
        // a 320x320 working surface. Reject other sizes rather than infer padding.
        demand(width==320 && height==144,"320x144 museum fixture required");
        std::vector<float> color(size_t(width)*height*3);
        std::ifstream in(job.at("color").get<std::string>(),std::ios::binary);
        demand(bool(in.read(reinterpret_cast<char*>(color.data()),color.size()*4)),"Color fixture unavailable");
        FrameState frame(width,height,320,320,4);
        std::vector<float> prior,post;std::vector<uint8_t> valid,post_valid;
        frame.warp(nullptr,nullptr,nullptr,false,prior,post,valid,post_valid);
        auto controls=Controls::from(job.at("metadata").at("controls"));
        auto features=frame.features(color.data(),controls,prior,valid);
        Model model(phoenix_model::WeightSource::config(cfg.at("weight_source")));
        Graph cpu_graph(*cpu,model,false,512,false,false),gpu_graph(*gpu,model,false,512,false,false);
        std::array<unsigned,8> repeats=job.at("metadata").at("repeats");
        cpu_graph.build(320,320,repeats);gpu_graph.build(320,320,repeats);
        model.discard_decoded();
        cpu->upload({cpu_graph.input.ptr(),features.size()*2,cpu_graph.input.memory->token},features.data(),features.size()*2);
        gpu->upload({gpu_graph.input.ptr(),features.size()*2,gpu_graph.input.memory->token},features.data(),features.size()*2);
        size_t offset=0;
        std::vector<Event> events;
        for(size_t i=0;i<cpu_graph.checkpoints.size();++i) {
            auto a=cpu_graph.checkpoints[i],b=gpu_graph.checkpoints[i];
            demand(a.after==b.after && a.bytes==b.bytes,"Checkpoint graph mismatch");
            if(a.after>offset) {
              if(std::getenv("PHX_COMMAND_PROBE")) {
                for(size_t index=offset;index<a.after;++index) {
                    auto ca=cpu_graph.commands[index],cb=gpu_graph.commands[index];
                    cpu->execute({ca},events);gpu->execute({cb},events);
                    size_t count=ca.count;
                    auto op=ca.operation;
                    if(op.family==phx::RowReduction)
                        count*=((op.flags&255u)==0u?32u:(op.left_batch_stride?op.left_batch_stride:op.column_count));
                    unsigned element=(op.flags&phx::StoreHalf)?2:4;
                    size_t bytes=count*element;
                    std::vector<uint8_t> x(bytes),y(bytes);
                    cpu->download({op.output,bytes,cpu_graph.input.memory->token},x.data(),bytes);
                    gpu->download({cb.operation.output,bytes,gpu_graph.input.memory->token},y.data(),bytes);
                    if(x!=y) {
                        size_t first=0;while(first<bytes&&x[first]==y[first])++first;
                        uint32_t xb=0,yb=0;
                        std::memcpy(&xb,x.data()+first/element*element,element);
                        std::memcpy(&yb,y.data()+first/element*element,element);
                        std::cerr<<"Command mismatch "<<index<<" family "<<op.family<<" flags "<<std::hex<<op.flags<<" CPU "<<xb<<" GPU "<<yb<<std::dec<<" element "<<first/element<<" shape "<<op.row_count<<","<<op.column_count<<","<<op.reduction_count<<" batches "<<op.batch_count<<std::endl;
                        std::cerr<<std::endl;
                        return 1;
                    }
                }
              } else {
                cpu->execute({cpu_graph.commands.begin()+offset,cpu_graph.commands.begin()+a.after},events);
                gpu->execute({gpu_graph.commands.begin()+offset,gpu_graph.commands.begin()+b.after},events);
              }
            }
            std::vector<uint8_t> expected(a.bytes),actual(b.bytes);
            cpu->download({reinterpret_cast<void*>(uintptr_t(a.allocation.address)+a.offset),a.bytes,a.allocation.token},expected.data(),a.bytes);
            gpu->download({reinterpret_cast<void*>(uintptr_t(b.allocation.address)+b.offset),b.bytes,b.allocation.token},actual.data(),b.bytes);
            if(actual!=expected) {
                size_t first=0,different=0;while(first<a.bytes&&actual[first]==expected[first])++first;
                for(size_t x=0;x<a.bytes;++x)different+=actual[x]!=expected[x];
                std::cerr<<"Mismatch "<<a.name<<" commands "<<offset<<".."<<a.after<<" first byte "<<first<<" different bytes "<<different<<" / "<<a.bytes<<std::endl;
                return 1;
            }
            std::cout<<a.name<<" exact ("<<a.bytes<<" bytes)"<<std::endl;
            offset=a.after;
        }
        std::cout<<"All model checkpoints match CPU byte for byte"<<std::endl;
    } catch(const std::exception& e) { std::cerr<<e.what()<<std::endl;return 1; }
}
