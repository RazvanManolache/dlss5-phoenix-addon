// GPU-resident public-API benchmark. Inputs/readback/status encoding are outside
// each timed call. No game, NVIDIA model DLL code, or reduced graph is executed.
#include "phoenix_native.h"
#include "../../src/native/runtime/texture_binding.h"
#include "../../third_party/nlohmann/json.hpp"
#include <bcrypt.h>
#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <numeric>
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
static std::string hash(const std::vector<uint8_t>& bytes){
    uint8_t h[32];char s[65]{};
    require(BCryptHash(BCRYPT_SHA256_ALG_HANDLE,nullptr,0,const_cast<PUCHAR>(bytes.data()),ULONG(bytes.size()),h,32)>=0,"SHA256");
    for(unsigned i=0;i<32;++i)sprintf_s(s+2*i,3,"%02x",unsigned(h[i]));return s;
}
static Json stats(std::vector<double> v){
    require(!v.empty(),"No timing samples");std::sort(v.begin(),v.end());
    auto q=[&](double x){return v[std::min(v.size()-1,size_t(x*(v.size()-1)))];};
    return {{"samples",v.size()},{"mean_ms",std::accumulate(v.begin(),v.end(),0.)/v.size()},
        {"median_ms",q(.5)},{"p95_ms",q(.95)},{"min_ms",v.front()},{"max_ms",v.back()}};
}
struct Api{
    HMODULE module=nullptr;
    decltype(&phoenix_native_create) create;
    decltype(&phoenix_native_destroy) destroy;
    decltype(&phoenix_native_device) device;
    decltype(&phoenix_native_status) status;
    decltype(&phoenix_native_error) error;
    decltype(&phoenix_native_frame_d3d12) frame;
    decltype(&phoenix_native_frame_d3d12_v2) advanced;
    explicit Api(const std::filesystem::path& p){
        module=LoadLibraryExW(p.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        require(module!=nullptr,"Load benchmark DLL");
#define LOAD(name,exported) name=reinterpret_cast<decltype(name)>(GetProcAddress(module,exported))
        LOAD(create,"phoenix_native_create");LOAD(destroy,"phoenix_native_destroy");LOAD(device,"phoenix_native_device");
        LOAD(status,"phoenix_native_status");LOAD(error,"phoenix_native_error");LOAD(frame,"phoenix_native_frame_d3d12");
        LOAD(advanced,"phoenix_native_frame_d3d12_v2");
#undef LOAD
        require(create&&destroy&&device&&status&&error&&frame,"Missing native exports");
    }
    ~Api(){if(module)FreeLibrary(module);}
    Json report(PhoenixNativeSession* session){
        std::string value(status(session,nullptr,0),'\0');status(session,value.data(),value.size());return Json::parse(value.c_str());
    }
};
struct Harness{
    unsigned w,h;ComPtr<ID3D12Device> device;ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;ComPtr<ID3D12GraphicsCommandList> command;ComPtr<ID3D12Fence> fence;
    uint64_t serial=0;
    ComPtr<ID3D12Resource> color,motion,output;
    Harness(Api& api,PhoenixNativeSession* session,unsigned width,unsigned height):w(width),h(height){
        require(api.device(session,&device,&queue)==0,api.error());
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
        check(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&command)));
        check(command->Close());check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
        color=make(DXGI_FORMAT_R16G16B16A16_FLOAT,8,0);
        motion=make(DXGI_FORMAT_R16G16_FLOAT,4,1);
        output=make(DXGI_FORMAT_R16G16B16A16_FLOAT,8,1);
    }
    void submit(std::function<void()> record){
        check(allocator->Reset());check(command->Reset(allocator.Get(),nullptr));record();check(command->Close());
        ID3D12CommandList* lists[]={command.Get()};queue->ExecuteCommandLists(1,lists);check(queue->Signal(fence.Get(),++serial));
        HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);require(event!=nullptr,"Fence event");
        check(fence->SetEventOnCompletion(serial,event));auto result=WaitForSingleObject(event,60000);CloseHandle(event);
        require(result==WAIT_OBJECT_0,"Harness fence timeout");
    }
    // kind: scene=0, zero=1, map=2, UI=3, alpha=4, backbuffer=5, field=6.
    ComPtr<ID3D12Resource> make(DXGI_FORMAT format,unsigned bpp,unsigned kind){
        D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;desc.Width=w;desc.Height=h;
        desc.DepthOrArraySize=1;desc.MipLevels=1;desc.Format=format;desc.SampleDesc.Count=1;desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;ComPtr<ID3D12Resource> image;
        check(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&image)));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT64 bytes=0;device->GetCopyableFootprints(&desc,0,1,0,&fp,nullptr,nullptr,&bytes);
        D3D12_RESOURCE_DESC buffer{};buffer.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;buffer.Width=bytes;buffer.Height=1;
        buffer.DepthOrArraySize=1;buffer.MipLevels=1;buffer.SampleDesc.Count=1;buffer.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        heap.Type=D3D12_HEAP_TYPE_UPLOAD;ComPtr<ID3D12Resource> upload;
        check(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&upload)));
        void* mapped=nullptr;check(upload->Map(0,nullptr,&mapped));
        for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
            auto* pixel=static_cast<uint8_t*>(mapped)+fp.Offset+size_t(y)*fp.Footprint.RowPitch+size_t(x)*bpp;
            const bool hud=y<h/10&&x>w/16&&x<w/3;
            if(kind==0||kind==5){uint16_t values[]={uint16_t(0x3000+(x/8%16)*0x40),uint16_t(0x3400+(y/8%8)*0x40),uint16_t(0x3800+((x+y)/16%8)*0x20),0x3c00};
                if(kind==5&&hud)values[0]=values[1]=values[2]=0x3c00;std::memcpy(pixel,values,8);}
            else if(kind==1)std::memset(pixel,0,bpp);
            else {float v[4]={};
                if(kind==2){v[0]=x<w/3?0.f:1.f;v[1]=x<w/2?1.f:.4f;v[2]=x<w/2?1.f:.25f;v[3]=1;}
                if(kind==3)v[0]=v[1]=v[2]=v[3]=hud?1.f:0.f;
                if(kind==4)v[0]=hud?1.f:0.f;
                if(kind==6){v[0]=-1.f/w;v[2]=1.f/w;}
                std::memcpy(pixel,v,bpp);}
        }
        upload->Unmap(0,nullptr);submit([&]{D3D12_TEXTURE_COPY_LOCATION from{},to{};from.pResource=upload.Get();
            from.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;from.PlacedFootprint=fp;to.pResource=image.Get();to.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            command->CopyTextureRegion(&to,0,0,0,&from,nullptr);barrier(command.Get(),image.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);});
        return image;
    }
    std::vector<uint8_t> readback(){
        const auto desc=output->GetDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT64 bytes=0;
        device->GetCopyableFootprints(&desc,0,1,0,&fp,nullptr,nullptr,&bytes);
        D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;d.Width=bytes;d.Height=1;d.DepthOrArraySize=1;d.MipLevels=1;d.SampleDesc.Count=1;d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_READBACK;ComPtr<ID3D12Resource> read;
        check(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&read)));
        submit([&]{barrier(command.Get(),output.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION from{},to{};from.pResource=output.Get();from.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            to.pResource=read.Get();to.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;to.PlacedFootprint=fp;
            command->CopyTextureRegion(&to,0,0,0,&from,nullptr);barrier(command.Get(),output.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);});
        std::vector<uint8_t> pixels(size_t(w)*h*8);void* mapped=nullptr;check(read->Map(0,nullptr,&mapped));
        for(unsigned y=0;y<h;++y)std::memcpy(pixels.data()+size_t(y)*w*8,static_cast<uint8_t*>(mapped)+fp.Offset+size_t(y)*fp.Footprint.RowPitch,size_t(w)*8);
        read->Unmap(0,nullptr);return pixels;
    }
};
int main(int argc,char** argv){
    Json result={{"passed",false},{"profiles",Json::array()}};
    try{
        require(argc==5,"DLL CONFIG_JSON RESULT_JSON legacy|candidate");Json config;std::ifstream(argv[2])>>config;
        const double warm_seconds=config.value("benchmark_warmup_seconds",.75),
            measure_seconds=config.value("benchmark_measured_seconds",2.0),
            cap_seconds=config.value("benchmark_cap_seconds",15.0);
        require(warm_seconds>0&&measure_seconds>0&&cap_seconds>=measure_seconds,"Invalid benchmark windows");
        Api api(std::filesystem::absolute(argv[1]));auto started=Clock::now();
        auto* session=api.create(config.dump().c_str(),nullptr,nullptr);require(session!=nullptr,api.error());
        auto destroy=[&](PhoenixNativeSession* s){require(api.destroy(s)==0,api.error());};
        std::unique_ptr<PhoenixNativeSession,decltype(destroy)> owned(session,destroy);
        result["create_ms"]=elapsed(started)*1000;
        const unsigned w=config.at("width"),h=config.at("height");Harness harness(api,session,w,h);
        std::vector<std::string> profiles={"default"};
        if(std::string(argv[4])=="candidate")profiles={"default","advanced_default","natural","cinematic","control_map","ui_overlay","ui_alpha_backbuffer","distortion","combined","default_after_features"};
        if(config.contains("benchmark_profiles"))profiles=config.at("benchmark_profiles").get<std::vector<std::string>>();
        result.update({{"width",w},{"height",h},{"gpu_resident",true},{"passes",1},{"repeats",{1,1,1,1,1,1,1,1}},
            {"warmup_seconds",warm_seconds},{"measured_seconds_min",measure_seconds},{"measured_seconds_cap",cap_seconds},{"minimum_samples",12},
            {"gpu_timing_scope","model + presentation/unwarp/output copy; excludes canonicalization/preparation"},
            {"wall_scope","complete synchronous GPU-texture API call, includes input preparation; excludes producer upload/readback/status parsing"}});
        for(const auto& name:profiles){
            PhoenixNativeFrameV2 request{};request.struct_bytes=sizeof(request);request.version=2;
            request.controls={sizeof(PhoenixNativeControls),1,1,1,1,1,-1,1,0};request.intensity=1;
            const auto state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            request.color={harness.color.Get(),state,0};request.motion={harness.motion.Get(),state,0};request.output={harness.output.Get(),state,0};
            std::vector<ComPtr<ID3D12Resource>> optional;
            auto add=[&](PhoenixNativeTextureV2& target,DXGI_FORMAT format,unsigned bytes,unsigned kind){optional.push_back(harness.make(format,bytes,kind));target={optional.back().Get(),state,0};};
            if(name=="natural"||name=="combined")request.style=1;
            if(name=="cinematic")request.style=2;
            if(name=="control_map"||name=="combined")add(request.control_mask,DXGI_FORMAT_R32G32B32A32_FLOAT,16,2);
            if(name=="ui_overlay"||name=="combined")add(request.ui,DXGI_FORMAT_R32G32B32A32_FLOAT,16,3);
            if(name=="ui_alpha_backbuffer"){add(request.ui_alpha,DXGI_FORMAT_R32_FLOAT,4,4);add(request.backbuffer,DXGI_FORMAT_R16G16B16A16_FLOAT,8,5);}
            if(name=="distortion"||name=="combined")add(request.distortion,DXGI_FORMAT_R32G32B32A32_FLOAT,16,6);
            const bool advanced=name!="default"&&name!="default_after_features";
            auto call=[&]{auto t=Clock::now();const int code=advanced?api.advanced(session,&request):api.frame(session,harness.color.Get(),state,harness.motion.Get(),state,harness.output.Get(),state,&request.controls);
                require(code==0,api.error());return elapsed(t)*1000;};
            const double first=call();request.controls.reset=0;
            unsigned warm=0;auto t=Clock::now();while(warm<4||elapsed(t)<warm_seconds){call();++warm;}
            std::vector<double> walls,gpus;t=Clock::now();
            while((walls.size()<12||elapsed(t)<measure_seconds)&&elapsed(t)<cap_seconds){walls.push_back(call());gpus.push_back(api.report(session).at("gpu_processing_ms"));}
            const double duration=elapsed(t);const auto status=api.report(session);
            request.controls.reset=1;call();auto pixels=harness.readback();
            for(size_t i=0;i<pixels.size();i+=2){uint16_t half;std::memcpy(&half,pixels.data()+i,2);require((half&0x7c00)!=0x7c00,"Nonfinite output");}
            Json row={{"profile",name},{"first_call_ms",first},{"warmup_frames",warm},{"measured_duration_seconds",duration},
                {"wall",stats(walls)},{"gpu",stats(gpus)},{"wall_samples_ms",walls},{"gpu_samples_ms",gpus},
                {"finite_reset_output",true},{"reset_output_sha256",hash(pixels)},{"runtime_status",status}};
            result["profiles"].push_back(row);std::ofstream(argv[3])<<result.dump(2)<<'\n';
            std::cout<<name<<" "<<w<<"x"<<h<<" GPU "<<row["gpu"]["median_ms"]<<" ms wall "<<row["wall"]["median_ms"]<<" ms\n"<<std::flush;
        }
        owned.reset();result["passed"]=true;std::ofstream(argv[3])<<result.dump(2)<<'\n';return 0;
    }catch(const std::exception& e){result["error"]=e.what();if(argc>=4)std::ofstream(argv[3])<<result.dump(2)<<'\n';std::cerr<<e.what()<<'\n';return 1;}
}
