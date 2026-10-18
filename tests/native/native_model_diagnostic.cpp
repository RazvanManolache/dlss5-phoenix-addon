// Explicit model/GPU diagnostic, deliberately not an automatic CTest case.
// Include the real standalone implementation so prefix execution and snapshots
// exercise the same ownership, descriptors, dispatch and host transfers.
#include "../../native_runtime/native_api.cpp"
#include <fstream>
#include <iostream>

namespace phoenix_native {
struct NetworkDiagnostics {
    static ID3D12Resource* workspace(Network& network) { return network.workspace_.Get(); }
    static void replace(Network& network,const Json& plan,const std::string& name,
                        const std::filesystem::path& binary,const std::string& entry) {
        std::ifstream stream(binary,std::ios::binary);
        std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(stream),std::istreambuf_iterator<char>()};
        require(!bytes.empty(),"Empty diagnostic kernel override");
        auto module=std::make_unique<phoenix_kernel::Module>(network.device_.Get(),network.api_.modules,network.counters);
        nv(network.api_.modules.create_module(network.device_.Get(),bytes.data(),NvU32(bytes.size()),&module->module));
        nv(network.api_.modules.create_function(network.device_.Get(),module->module,entry.c_str(),&module->function));
        for(auto& program:network.programs_)for(unsigned i=0;i<program.size();++i)
            if(plan.at("launches").at(i+1).at("name")==name)program[i].launch.hFunction=module->function;
        network.modules_.push_back(std::move(module));
    }
    static void prefix(Network& network, unsigned limit, bool serialize) {
        require(limit && limit <= 154, "Diagnostic prefix outside 1..154");
        if (serialize) {
            network.group_ends_.clear();
            for (unsigned i = 1; i <= limit; ++i) network.group_ends_.push_back(i);
            for (auto& program : network.programs_)
                for (auto& step : program) step.barrier_after = true;
        } else {
            std::erase_if(network.group_ends_, [=](unsigned end) { return end > limit; });
            if (network.group_ends_.empty() || network.group_ends_.back() != limit)
                network.group_ends_.push_back(limit);
            for (auto& program : network.programs_) program[limit - 1].barrier_after = true;
        }
    }
};
}
namespace {
struct Capture {
    ID3D12Resource* workspace;
    ComPtr<ID3D12Resource> readback, stamps, counter_readback;
    uint64_t counter_bytes=0;
    struct Slot { unsigned end; uint64_t offset, bytes; std::string tensor; uint64_t readback_offset=0; };
    std::vector<Slot> slots;
    inline static thread_local Capture* current = nullptr;
    Capture(PhoenixNativeSession& session, const Json& plan,const Json& config) : workspace(phoenix_native::NetworkDiagnostics::workspace(*session.network)) {
        counter_bytes=UINT64_MAX;
        for(const auto& allocation:plan.at("allocation"))counter_bytes=std::min(counter_bytes,allocation.at("offset").get<uint64_t>());
        counter_readback=phoenix_native::buffer(session.device.Get(),counter_bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        uint64_t size=0;
        for (const auto end : session.network->group_ends()) {
            Slot slot{end, 0, 0, "output texture"};
            const auto& source = plan.at("launches").at(end);
            for (const auto& binding : source.at("relocations")) {
                if (binding.at("role") != "output" || binding.at("arena") != "workspace" || binding.at("relative_offset").is_null()) continue;
                slot.offset = binding.at("relative_offset");
                for (const auto& allocation : plan.at("allocation"))
                    if (allocation.at("offset") == slot.offset) {
                        // Multiple aliased tensors may have the same offset;
                        // choose the currently live tensor.
                        const auto first=allocation.at("first_group").get<unsigned>();
                        const auto last=allocation.at("last_group").get<unsigned>();
                        if (first <= end && end <= last) {
                            slot.bytes=std::min<uint64_t>(config.value("diagnostic_capture_bytes",uint64_t(65536)),allocation.at("bytes").get<uint64_t>());
                            slot.tensor=allocation.at("tensor");
                        }
                    }
                break;
            }
            if(config.contains("diagnostic_capture_launches")) {
                const auto& wanted=config.at("diagnostic_capture_launches");
                if(std::find(wanted.begin(),wanted.end(),Json(end))==wanted.end())slot.bytes=0;
            }
            slot.readback_offset=size;size+=phoenix_generated::align(slot.bytes+256,256);
            slots.push_back(slot);
        }
        readback=phoenix_native::buffer(session.device.Get(),size,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        stamps=phoenix_native::buffer(session.device.Get(),size,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        void* data=nullptr;D3D12_RANGE empty{};phoenix_native::hr(stamps->Map(0,&empty,&data));
        std::memset(data,0,size_t(size));
        for(unsigned i=0;i<slots.size();++i)static_cast<uint32_t*>(data)[slots[i].readback_offset/4]=0xab000000u+slots[i].end;
        stamps->Unmap(0,nullptr);
    }
    static void mark(ID3D12GraphicsCommandList* cmd,unsigned group) {
        auto& self=*current;const auto& slot=self.slots.at(group);
        if(slot.bytes) {
            phoenix_native::transition(cmd,self.workspace,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
            cmd->CopyBufferRegion(self.readback.Get(),slot.readback_offset+256,self.workspace,slot.offset,slot.bytes);
            phoenix_native::transition(cmd,self.workspace,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        cmd->CopyBufferRegion(self.readback.Get(),slot.readback_offset,self.stamps.Get(),slot.readback_offset,4);
        if(group+1==self.slots.size()) {
            phoenix_native::transition(cmd,self.workspace,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
            cmd->CopyBufferRegion(self.counter_readback.Get(),0,self.workspace,0,self.counter_bytes);
            phoenix_native::transition(cmd,self.workspace,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }
    Json timeout_flags(const Json& plan) {
        uint32_t* data=nullptr;D3D12_RANGE range{0,size_t(counter_bytes)};
        phoenix_native::hr(counter_readback->Map(0,&range,reinterpret_cast<void**>(&data)));
        Json result=Json::array();
        for(uint64_t i=0;i<counter_bytes/4;++i)if(data[i]==0x5eaddeadu) {
            uint64_t base=0;std::string role;unsigned index=0;
            for(const auto& launch:plan.at("launches"))for(const auto& b:launch.at("relocations"))
                if(b.at("arena")=="workspace"&&!b.at("relative_offset").is_null()&&
                   b.at("role").get<std::string>().find("counter")!=std::string::npos&&
                   b.at("relative_offset").get<uint64_t>()<=i*4&&b.at("relative_offset").get<uint64_t>()>=base) {
                    base=b.at("relative_offset");role=b.at("role");index=launch.at("index");
                }
            result.push_back({{"offset",i*4},{"nearest_counter_launch",index},{"role",role},{"word_offset",(i*4-base)/4}});
        }
        D3D12_RANGE empty{};counter_readback->Unmap(0,&empty);return result;
    }
    std::vector<std::vector<uint8_t>> values() {
        uint8_t* data=nullptr;D3D12_RANGE range{0,size_t(readback->GetDesc().Width)};
        phoenix_native::hr(readback->Map(0,&range,reinterpret_cast<void**>(&data)));
        std::vector<std::vector<uint8_t>> result;
        for(const auto& slot:slots) result.emplace_back(data+slot.readback_offset+256,data+slot.readback_offset+256+slot.bytes);
        D3D12_RANGE empty{};readback->Unmap(0,&empty);return result;
    }
};
}
int main(int argc,char** argv) {
    Json report={{"passed",false},{"diagnostic_only",true}};
    if(argc!=3){std::cerr<<"CONFIG_JSON REPORT_JSON\n";return 2;}
    try {
        Json config;std::ifstream(argv[1])>>config;
        const unsigned w=config.at("width"),h=config.at("height");
        const unsigned limit=config.value("diagnostic_launch_limit",154u);
        const bool serialize=config.value("diagnostic_full_barriers",false);
        PhoenixNativeSession session(config,nullptr,nullptr);
        phoenix_bundle::Bundle bundle(std::filesystem::u8path(config.at("bundle_path").get<std::string>()),file_digest,
            phoenix_model::WeightSource::config(config.at("weight_source")));
        auto plan=bundle.select_frame(w,h,[](const Json&){});
        if(config.contains("diagnostic_kernel_override"))for(const auto& entry:config.at("diagnostic_kernel_override"))
            phoenix_native::NetworkDiagnostics::replace(*session.network,plan,entry.at("name"),
                std::filesystem::u8path(entry.at("binary").get<std::string>()),entry.at("entry"));
        phoenix_native::NetworkDiagnostics::prefix(*session.network,limit,serialize);
        Capture capture(session,plan,config);
        Capture::current=&capture;phoenix_native::group_marker=&Capture::mark;
        std::vector<uint16_t> color(size_t(w)*h*4),motion(size_t(w)*h*2),output(color.size());
        for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x) {
            const size_t p=(size_t(y)*w+x)*4;
            color[p]=0x3000+(x/8%16)*0x40;color[p+1]=0x3400+(y/8%8)*0x40;
            color[p+2]=0x3800+((x+y)/16%8)*0x20;color[p+3]=0x3c00;
        }
        auto evaluate=[&](bool reset) {
            PhoenixNativeControls options{};options.struct_bytes=sizeof(options);options.reset=reset;
            options.motion_scale_x=options.motion_scale_y=options.local_tone=options.local_structure=1;
            options.skin_structure=-1;options.auto_mask=1;
            if(limit==154) {
                if(phoenix_native_frame_host(&session,color.data(),color.size()*2,w*8,motion.data(),motion.size()*2,w*4,
                    output.data(),output.size()*2,w*8,&options))throw std::runtime_error(phoenix_native_error());
            } else {
                session.prepare_plane(session.host_color,DXGI_FORMAT_R16G16B16A16_FLOAT,false);
                session.prepare_plane(session.host_motion,DXGI_FORMAT_R16G16_FLOAT,false);
                session.begin();session.upload(session.host_color,color.data(),w*8,w*8);
                session.upload(session.host_motion,motion.data(),w*4,w*4);session.submit();session.wait(session.submitted);
                session.begin();const auto state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                session.copy_input(session.host_color.image.Get(),state,session.input_color.Get(),false);
                session.copy_input(session.host_motion.image.Get(),state,session.input_motion.Get(),true);
                auto controls=phoenix_addon::resolve_controls({},w,h,true,false,0);
                phoenix_addon::FrameDispatch::Scope scope(*session.dispatch,session.command.Get(),controls);
                session.network->record(session.command.Get(),true,0,session.queue.Get());
                session.submit();session.network->commit(session.command.Get());session.wait(session.submitted);
            }
        };
        std::vector<std::vector<uint8_t>> first;
        report["frames"]=Json::array();
        const unsigned frames=config.value("diagnostic_reset_frames",3u);
        for(unsigned frame=0;frame<frames;++frame) {
            if(frame)for(unsigned i=0;i<config.value("diagnostic_continuations",0u);++i)evaluate(false);
            evaluate(true);auto values=capture.values();Json comparisons=Json::array();
            for(unsigned group=0;group<capture.slots.size();++group) {
                const auto& slot=capture.slots[group];size_t changed=0;
                if(frame)for(size_t b=0;b<values[group].size();++b)changed+=values[group][b]!=first[group][b];
                comparisons.push_back({{"last_launch",slot.end},{"kernel",plan.at("launches").at(slot.end).at("name")},
                    {"tensor",slot.tensor},{"bytes",slot.bytes},{"changed_bytes",changed},
                    {"sha256",phoenix_model::weight_digest(values[group].data(),values[group].size())}});
            }
            if(!frame)first=values;
            report["frames"].push_back(comparisons);
            std::cout<<"reset "<<frame<<" completed\n"<<std::flush;
        }
        report["passed"]=true;report["launch_limit"]=limit;report["serialized"]=serialize;
        report["diagnostic_timeout_flags"]=capture.timeout_flags(plan);
        if(config.contains("diagnostic_counters_file")) {
            void* data=nullptr;D3D12_RANGE range{0,size_t(capture.counter_bytes)};
            phoenix_native::hr(capture.counter_readback->Map(0,&range,&data));
            std::ofstream stream(config.at("diagnostic_counters_file").get<std::string>(),std::ios::binary);
            stream.write(static_cast<const char*>(data),std::streamsize(capture.counter_bytes));
            D3D12_RANGE empty{};capture.counter_readback->Unmap(0,&empty);
            report["plan"]=plan;
        }
        phoenix_native::group_marker=nullptr;Capture::current=nullptr;
    } catch(const std::exception& e){report["error"]=e.what();std::cerr<<e.what()<<'\n';}
    std::ofstream(argv[2])<<report.dump(2)<<'\n';return report.at("passed").get<bool>()?0:1;
}
