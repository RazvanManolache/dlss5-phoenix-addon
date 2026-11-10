// Explicit hardware contract, never registered as an automatic GPU test.
#include "../../native_runtime/phoenix.h"
#include "../../native_runtime/phoenix_cuda_d3d12.h"
#include "../../third_party/nlohmann/json.hpp"
#include <cuda.h>
#include <dxgi1_6.h>
#include <d3d12sdklayers.h>
#include <DirectXPackedVector.h>
#include <wrl/client.h>
#include <fstream>
#include <filesystem>
#include <vector>
#include <iostream>
#include <cmath>

using Microsoft::WRL::ComPtr;using Json=nlohmann::json;namespace fs=std::filesystem;
void demand(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void hr(HRESULT value){demand(SUCCEEDED(value),"D3D12 contract operation failed");}
struct Gpu {
    ComPtr<ID3D12Device> device;ComPtr<ID3D12CommandQueue> queue;ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> command;ComPtr<ID3D12Fence> fence;ComPtr<ID3D12InfoQueue> messages;
    uint64_t serial{};
    explicit Gpu(CUdevice cuda){
        ComPtr<ID3D12Debug> debug;if(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))debug->EnableDebugLayer();
        char luid[8]{};unsigned mask{};demand(cuDeviceGetLuid(luid,&mask,cuda)==CUDA_SUCCESS,"No CUDA LUID");
        ComPtr<IDXGIFactory4> factory;hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));ComPtr<IDXGIAdapter1> selected;
        for(unsigned i=0;;++i){ComPtr<IDXGIAdapter1> candidate;if(factory->EnumAdapters1(i,&candidate)==DXGI_ERROR_NOT_FOUND)break;
            DXGI_ADAPTER_DESC1 desc{};hr(candidate->GetDesc1(&desc));if(!memcmp(luid,&desc.AdapterLuid,8)){selected=candidate;break;}}
        demand(bool(selected),"Selected CUDA adapter has no DXGI device");hr(D3D12CreateDevice(selected.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)));
        device.As(&messages);D3D12_COMMAND_QUEUE_DESC q{};hr(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)));
        hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
        hr(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&command)));hr(command->Close());
        hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
    }
    void begin(){hr(allocator->Reset());hr(command->Reset(allocator.Get(),nullptr));}
    void end(){hr(command->Close());ID3D12CommandList* commands[]{command.Get()};queue->ExecuteCommandLists(1,commands);hr(queue->Signal(fence.Get(),++serial));
        HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);demand(event,"Cannot create completion event");
        hr(fence->SetEventOnCompletion(serial,event));const auto result=WaitForSingleObject(event,30000);CloseHandle(event);demand(result==WAIT_OBJECT_0,"GPU contract timed out");}
    void barrier(ID3D12Resource* image,D3D12_RESOURCE_STATES from,D3D12_RESOURCE_STATES to){D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;b.Transition={image,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,from,to};command->ResourceBarrier(1,&b);}
    ComPtr<ID3D12Resource> texture(unsigned w,unsigned h,DXGI_FORMAT format){
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width=w;d.Height=h;d.DepthOrArraySize=d.MipLevels=1;d.Format=format;d.SampleDesc.Count=1;ComPtr<ID3D12Resource> r;
        hr(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&r)));return r;
    }
    std::vector<uint16_t> transfer(ID3D12Resource* image,unsigned channels,const std::vector<uint16_t>* upload=nullptr){
        const auto desc=image->GetDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};uint64_t bytes{};
        device->GetCopyableFootprints(&desc,0,1,0,&footprint,nullptr,nullptr,&bytes);
        D3D12_HEAP_PROPERTIES heap{};heap.Type=upload?D3D12_HEAP_TYPE_UPLOAD:D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer{};buffer.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;buffer.Width=bytes;buffer.Height=buffer.DepthOrArraySize=buffer.MipLevels=1;
        buffer.SampleDesc.Count=1;buffer.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;ComPtr<ID3D12Resource> staging;
        hr(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,upload?D3D12_RESOURCE_STATE_GENERIC_READ:D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&staging)));
        void* mapped{};D3D12_RANGE empty{};
        if(upload){hr(staging->Map(0,&empty,&mapped));for(unsigned y=0;y<desc.Height;++y)memcpy(static_cast<char*>(mapped)+footprint.Offset+size_t(y)*footprint.Footprint.RowPitch,upload->data()+size_t(y)*desc.Width*channels,size_t(desc.Width)*channels*2);staging->Unmap(0,nullptr);}
        begin();const auto state=upload?D3D12_RESOURCE_STATE_COPY_DEST:D3D12_RESOURCE_STATE_COPY_SOURCE;barrier(image,D3D12_RESOURCE_STATE_COMMON,state);
        D3D12_TEXTURE_COPY_LOCATION texture{image,D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0},linear{staging.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,footprint};
        if(upload)command->CopyTextureRegion(&texture,0,0,0,&linear,nullptr);else command->CopyTextureRegion(&linear,0,0,0,&texture,nullptr);
        barrier(image,state,D3D12_RESOURCE_STATE_COMMON);end();
        std::vector<uint16_t> output(size_t(desc.Width)*desc.Height*channels);
        if(!upload){D3D12_RANGE range{0,size_t(bytes)};hr(staging->Map(0,&range,&mapped));for(unsigned y=0;y<desc.Height;++y)memcpy(output.data()+size_t(y)*desc.Width*channels,static_cast<char*>(mapped)+footprint.Offset+size_t(y)*footprint.Footprint.RowPitch,size_t(desc.Width)*channels*2);staging->Unmap(0,&empty);}
        return output;
    }
    void check_debug(){if(!messages)return;
        for(uint64_t i=0;i<messages->GetNumStoredMessagesAllowedByRetrievalFilter();++i){SIZE_T bytes{};messages->GetMessage(i,nullptr,&bytes);std::vector<uint8_t> data(bytes);auto* message=reinterpret_cast<D3D12_MESSAGE*>(data.data());hr(messages->GetMessage(i,message,&bytes));
            if(message->Severity<=D3D12_MESSAGE_SEVERITY_ERROR)throw std::runtime_error(message->pDescription);}
    }
};
int wmain(int argc,wchar_t** argv){
    try{
        demand(argc==3,"Usage: gpu-contract job.json report.json");Json job;{std::ifstream f{fs::path(argv[1])};f>>job;}
        const auto root=fs::u8path(job.at("runtime_root").get<std::string>());const unsigned w=job.at("width"),h=job.at("height");const size_t pixels=size_t(w)*h;
        auto config=job.at("config");config["runtime_root"]=root.string();
        demand(cuInit(0)==CUDA_SUCCESS,"CUDA initialization failed");CUdevice selected=-1;int count{};cuDeviceGetCount(&count);
        for(int i=0;i<count;++i){CUuuid uuid{};cuDeviceGetUuid(&uuid,i);std::string id="cuda:";for(unsigned char value:uuid.bytes){char hex[3];sprintf_s(hex,"%02x",value);id+=hex;}if(id==config.at("device").get<std::string>())selected=i;}
        demand(selected>=0,"CUDA UUID missing");Gpu gpu(selected);
        CUcontext caller{};demand(cuCtxCreate(&caller,nullptr,0,selected)==CUDA_SUCCESS,"Cannot create caller context");
        HMODULE module=LoadLibraryExW((root/L"phoenix_native_cuda.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);demand(module,"Cannot load native CUDA");
        const auto create=reinterpret_cast<decltype(&phoenix_runtime_create)>(GetProcAddress(module,"phoenix_runtime_create"));
        const auto destroy=reinterpret_cast<decltype(&phoenix_runtime_destroy)>(GetProcAddress(module,"phoenix_runtime_destroy"));
        const auto cpu_frame=reinterpret_cast<decltype(&phoenix_runtime_frame_v2)>(GetProcAddress(module,"phoenix_runtime_frame_v2"));
        const auto gpu_frame=reinterpret_cast<decltype(&phoenix_runtime_frame_d3d12)>(GetProcAddress(module,"phoenix_runtime_frame_d3d12"));
        const auto status=reinterpret_cast<decltype(&phoenix_runtime_status)>(GetProcAddress(module,"phoenix_runtime_status"));
        const auto error=reinterpret_cast<decltype(&phoenix_runtime_error)>(GetProcAddress(module,"phoenix_runtime_error"));
        const auto replace=reinterpret_cast<decltype(&phoenix_runtime_replace_model)>(GetProcAddress(module,"phoenix_runtime_replace_model"));
        demand(create&&destroy&&cpu_frame&&gpu_frame&&status&&error&&replace,"GPU API exports missing");
        auto own=[&](void* p){demand(p,error());return std::unique_ptr<void,decltype(destroy)>(p,destroy);};
        auto cpu=own(create(config.dump().c_str(),w,h)),device=own(create(config.dump().c_str(),w,h));
        auto preserved=[&]{CUcontext current{};demand(cuCtxGetCurrent(&current)==CUDA_SUCCESS&&current==caller,"Caller context changed");};preserved();
        auto state=[&]{std::string text(status(device.get(),nullptr,0),'\0');status(device.get(),text.data(),text.size());return Json::parse(text.c_str());};
        std::vector<float> original(pixels*3);{std::ifstream f(fs::u8path(job.at("input").get<std::string>()),std::ios::binary);f.read(reinterpret_cast<char*>(original.data()),original.size()*4);demand(bool(f),"Cannot read contract input");}
        std::vector<uint16_t> rgba(pixels*4),motion(pixels*2);std::vector<float> rgb(pixels*3),vectors(pixels*2),out(pixels*3);
        for(size_t p=0;p<pixels;++p){for(unsigned c=0;c<3;++c){rgba[p*4+c]=DirectX::PackedVector::XMConvertFloatToHalf(original[p*3+c]);rgb[p*3+c]=DirectX::PackedVector::XMConvertHalfToFloat(rgba[p*4+c]);}rgba[p*4+3]=DirectX::PackedVector::XMConvertFloatToHalf(float(p%17)/16);}
        auto color=gpu.texture(w,h,DXGI_FORMAT_R16G16B16A16_FLOAT),mv=gpu.texture(w,h,DXGI_FORMAT_R16G16_FLOAT),output=gpu.texture(w,h,DXGI_FORMAT_R16G16B16A16_FLOAT);
        phoenix_cuda_d3d12_frame frame{};frame.struct_bytes=sizeof(frame);frame.version=1;frame.queue=gpu.queue.Get();frame.color=color.Get();frame.motion=mv.Get();frame.output=output.Get();frame.motion_scale_x=frame.motion_scale_y=1;
        phoenix_frame_v2 host{};host.struct_bytes=sizeof(host);host.version=2;
        host.color={rgb.data(),rgb.size()*4,0,w,h,3,0};host.motion={vectors.data(),vectors.size()*4,0,w,h,2,0};host.output={out.data(),out.size()*4,0,w,h,3,0};
        Json checks=Json::array();
        {
            ComPtr<IDXGIFactory4> factory;hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));ComPtr<IDXGIAdapter> adapter;hr(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
            ComPtr<ID3D12Device> warp;hr(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&warp)));
            ComPtr<ID3D12CommandQueue> queue;D3D12_COMMAND_QUEUE_DESC description{};hr(warp->CreateCommandQueue(&description,IID_PPV_ARGS(&queue)));
            auto foreign=frame;foreign.queue=queue.Get();demand(gpu_frame(device.get(),&foreign)==1,"Foreign adapter did not refuse GPU transport");preserved();
            demand(state().at("frames")==0,"Unavailable GPU transport advanced history");checks.push_back("foreign adapter refuses GPU transport before evaluation");
        }
        auto evaluate=[&](Json metadata){
            gpu.transfer(color.Get(),4,&rgba);gpu.transfer(mv.Get(),2,&motion);
            const auto text=metadata.dump();frame.metadata_json=host.metadata_json=text.c_str();
            demand(cpu_frame(cpu.get(),&host)==0,error());preserved();demand(gpu_frame(device.get(),&frame)==0,error());preserved();
            const auto actual=gpu.transfer(output.Get(),4);
            for(size_t p=0;p<pixels;++p){for(unsigned c=0;c<3;++c)if(actual[p*4+c]!=DirectX::PackedVector::XMConvertFloatToHalf(out[p*3+c]))throw std::runtime_error("GPU transport RGB mismatch at "+std::to_string(p)+" channel "+std::to_string(c));demand(actual[p*4+3]==rgba[p*4+3],"GPU transport changed alpha");}
        };
        evaluate({{"reset",true}});checks.push_back("reset RGB exact against bounded CPU transport; alpha retained");
        for(size_t p=0;p<pixels;++p){motion[p*2]=DirectX::PackedVector::XMConvertFloatToHalf(-1);vectors[p*2]=-1;}
        evaluate({{"reset",false}});checks.push_back("recurrent motion exact against bounded CPU transport");
        for(float scale:{.375f,-1.7f,2.25f}){
            frame.motion_scale_x=scale;frame.motion_scale_y=-scale;
            for(size_t p=0;p<pixels;++p){vectors[p*2]=DirectX::PackedVector::XMConvertHalfToFloat(motion[p*2])*scale;vectors[p*2+1]=0;}
            evaluate({{"reset",false}});
        }
        checks.push_back("changing nonunit motion scales preserve recurrent CPU arithmetic exactly");
        frame.motion_scale_x=frame.motion_scale_y=1;for(size_t p=0;p<pixels;++p)vectors[p*2]=-1;
        for(const auto& controls:std::vector<Json>{{{"local_tone",0.f},{"local_structure",0.f}},{{"local_tone",2.f},{"local_structure",2.f},{"auto_mask",true}},{{"intensity",0.f}},{{"intensity",.5f}},{{"style",1}},{{"style",2}},
            {{"intensity",2.f},{"style",1},{"local_tone",2.f},{"local_structure",2.f}},
            {{"intensity",2.f},{"style",2},{"local_tone",2.f},{"local_structure",2.f}}})evaluate({{"reset",true},{"controls",controls}});
        checks.push_back("conditioning, intensity and both authenticated styles exact");
        for(auto metadata:std::vector<Json>{{{"reset",true},{"transformer_blocks",4}},{{"reset",true},{"repeats",{1,0,2,1,0,1,1,1}}},{{"reset",true},{"repeats",{0,0,0,0,0,0,0,0}}},{{"reset",true},{"repeats",{4,4,4,4,4,4,4,4}}},{{"reset",true}}})evaluate(metadata);
        checks.push_back("truncation, bypass, maximum repetitions and recovery exact");
        demand(state().at("retained_residual_copies")==0,"Full schedule retained adjustment-only residual copies");
        checks.push_back("returning to full workload restores original copy-free graph");
        const auto before=gpu.transfer(output.Get(),4);const auto frames=state().at("frames");
        std::string text="{\"reset\":false}";frame.metadata_json=text.c_str();auto bad=frame;bad.version=99;
        demand(gpu_frame(device.get(),&bad)==-1,"Invalid GPU header accepted");preserved();bad=frame;bad.output=bad.color;demand(gpu_frame(device.get(),&bad)==-1,"GPU alias accepted");preserved();
        text="{\"passes\":2}";frame.metadata_json=text.c_str();demand(gpu_frame(device.get(),&frame)==-1,"GPU multipass leaf accepted");preserved();
        auto wrong=gpu.texture(w+1,h,DXGI_FORMAT_R16G16B16A16_FLOAT);bad=frame;bad.output=wrong.Get();text="{}";bad.metadata_json=text.c_str();demand(gpu_frame(device.get(),&bad)==-1,"Wrong GPU extent accepted");preserved();
        demand(gpu.transfer(output.Get(),4)==before&&state().at("frames")==frames,"Rejected GPU request changed completion or output");checks.push_back("invalid header, alias and schedule rejected atomically");
        const auto saved=rgba[0];rgba[0]=0x7e00;gpu.transfer(color.Get(),4,&rgba);text="{\"reset\":false}";frame.metadata_json=text.c_str();demand(gpu_frame(device.get(),&frame)==-1,"GPU NaN accepted");preserved();rgba[0]=saved;
        demand(gpu.transfer(output.Get(),4)==before&&state().at("frames")==frames,"Rejected GPU NaN wrote output or history");evaluate({{"reset",false}});checks.push_back("GPU validation rejection preserves following temporal output");
        {const auto completed=state().at("frames");const auto previous=gpu.transfer(output.Get(),4);const auto saved_motion=motion[0];
            motion[0]=DirectX::PackedVector::XMConvertFloatToHalf(2);gpu.transfer(mv.Get(),2,&motion);frame.motion_scale_x=3.4e38f;
            demand(gpu_frame(device.get(),&frame)==-1,"Overflowing scaled GPU motion accepted");preserved();
            demand(state().at("frames")==completed&&gpu.transfer(output.Get(),4)==previous,"Rejected scaled motion changed output or history");
            motion[0]=saved_motion;frame.motion_scale_x=1;evaluate({{"reset",false}});checks.push_back("scaled motion overflow rejected without changing following history");}
        for(const auto& repeats:std::vector<Json>{{1,1,1,1,1,1,1,1},{1,0,2,1,0,1,1,1},{0,0,0,0,0,0,0,0}}){
            evaluate({{"reset",true},{"repeats",repeats}});
            for(const bool bad_motion:{false,true}){
                const auto completed=state().at("frames"),previous_status=state();
                const auto previous=gpu.transfer(output.Get(),4);
                auto& plane=bad_motion?motion:rgba;const auto saved_value=plane[0];
                plane[0]=bad_motion?0x7c00:0x7e00;
                gpu.transfer((bad_motion?mv:color).Get(),bad_motion?2:4,&plane);
                const Json rejected_repeats=repeats==Json{1,1,1,1,1,1,1,1}?Json{1,0,2,1,0,1,1,1}:Json{1,1,1,1,1,1,1,1};
                text=Json{{"reset",false},{"repeats",rejected_repeats}}.dump();frame.metadata_json=text.c_str();
                demand(gpu_frame(device.get(),&frame)==-1,"Nonfinite GPU input accepted after schedule change");preserved();
                demand(state()==previous_status&&state().at("frames")==completed&&gpu.transfer(output.Get(),4)==previous,
                    "Rejected GPU input published status/history/output");
                plane[0]=saved_value;evaluate({{"reset",false},{"repeats",repeats}});
            }
        }
        checks.push_back("combined validation preserves exact continuation across full, repeated, bypass and rejected schedule changes");
        {text="{\"reset\":true}";host.metadata_json=text.c_str();demand(cpu_frame(cpu.get(),&host)==0,error());const auto expected=out;
            text="{\"reset\":false}";host.metadata_json=text.c_str();demand(cpu_frame(device.get(),&host)==0,error());preserved();
            demand(out==expected,"GPU to CPU transport transition did not reset history");demand(state().at("neural_host_transfer_bytes").get<uint64_t>()>0,"CPU transport retained zero byte telemetry");
            text="{\"reset\":true}";host.metadata_json=text.c_str();demand(cpu_frame(cpu.get(),&host)==0,error());
            text="{\"reset\":false}";frame.metadata_json=text.c_str();demand(gpu_frame(device.get(),&frame)==0,error());preserved();const auto actual=gpu.transfer(output.Get(),4);
            for(size_t p=0;p<pixels;++p)for(unsigned c=0;c<3;++c)demand(actual[p*4+c]==DirectX::PackedVector::XMConvertFloatToHalf(out[p*3+c]),"CPU to GPU transport transition did not reset history");
            checks.push_back("CPU and GPU transport switches reset safely and refresh byte telemetry");}
        demand(replace(cpu.get(),config.at("weight_source").dump().c_str())==0,error());demand(replace(device.get(),config.at("weight_source").dump().c_str())==0,error());preserved();evaluate({{"reset",false}});checks.push_back("model replacement rebuilds GPU bindings and resets history");
        const auto final=state();demand(final.at("pixel_transport")=="d3d12-cuda-shared-textures"&&final.at("neural_host_transfer_bytes")==0,"GPU transport was not used");gpu.check_debug();
        cpu.reset();device.reset();preserved();FreeLibrary(module);cuCtxDestroy(caller);
        std::ofstream report{fs::path(argv[2])};report<<Json{{"passed",true},{"checks",checks},{"final_status",final},{"debug_layer",bool(gpu.messages)}}.dump(2)<<'\n';
        std::cout<<"GPU texture transport: "<<checks.size()<<" contract groups passed\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
