// Explicit benchmark: fresh full graph versus full graph after workload edits.
#define wmain native_cuda_unused_contract_main
#include "native_cuda_d3d12_tests.cpp"
#undef wmain
#include "../../src/native/model/weight_source.h"
#include <chrono>
#include <algorithm>

int wmain(int argc,wchar_t** argv){
    try{
        demand(argc==3,"JOB.json REPORT.json");Json job;std::ifstream job_file{fs::path(argv[1])};job_file>>job;
        const auto root=fs::u8path(job.at("runtime_root").get<std::string>());
        auto config=job.at("config");config["runtime_root"]=root.string();
        const unsigned w=job.at("width"),h=job.at("height");const size_t pixels=size_t(w)*h;
        demand(cuInit(0)==CUDA_SUCCESS,"CUDA initialization failed");int count{};CUdevice selected=-1;cuDeviceGetCount(&count);
        for(int i=0;i<count;++i){CUuuid uuid{};cuDeviceGetUuid(&uuid,i);std::string id="cuda:";
            for(unsigned char value:uuid.bytes){char hex[3];sprintf_s(hex,"%02x",value);id+=hex;}
            if(id==config.at("device").get<std::string>())selected=i;}
        demand(selected>=0,"CUDA UUID missing");Gpu gpu(selected);
        const auto module=LoadLibraryExW((root/L"phoenix_native_cuda.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        demand(module,"Native CUDA DLL missing");
        const auto create=reinterpret_cast<decltype(&phoenix_runtime_create)>(GetProcAddress(module,"phoenix_runtime_create"));
        const auto destroy=reinterpret_cast<decltype(&phoenix_runtime_destroy)>(GetProcAddress(module,"phoenix_runtime_destroy"));
        const auto evaluate=reinterpret_cast<decltype(&phoenix_runtime_frame_d3d12)>(GetProcAddress(module,"phoenix_runtime_frame_d3d12"));
        const auto status=reinterpret_cast<decltype(&phoenix_runtime_status)>(GetProcAddress(module,"phoenix_runtime_status"));
        const auto error=reinterpret_cast<decltype(&phoenix_runtime_error)>(GetProcAddress(module,"phoenix_runtime_error"));
        demand(create&&destroy&&evaluate&&status&&error,"Native CUDA API missing");
        auto session=std::unique_ptr<void,decltype(destroy)>(create(config.dump().c_str(),w,h),destroy);demand(bool(session),error());
        auto state=[&]{std::string text(status(session.get(),nullptr,0),'\0');status(session.get(),text.data(),text.size());return Json::parse(text.c_str());};
        std::vector<float> rgb(pixels*3);std::ifstream input(fs::u8path(job.at("input").get<std::string>()),std::ios::binary);
        input.read(reinterpret_cast<char*>(rgb.data()),rgb.size()*4);demand(bool(input),"Input missing");
        std::vector<uint16_t> rgba(pixels*4),motion(pixels*2);
        for(size_t p=0;p<pixels;++p){for(unsigned c=0;c<3;++c)rgba[p*4+c]=DirectX::PackedVector::XMConvertFloatToHalf(rgb[p*3+c]);rgba[p*4+3]=0x3c00;}
        auto color=gpu.texture(w,h,DXGI_FORMAT_R16G16B16A16_FLOAT),mv=gpu.texture(w,h,DXGI_FORMAT_R16G16_FLOAT),output=gpu.texture(w,h,DXGI_FORMAT_R16G16B16A16_FLOAT);
        gpu.transfer(color.Get(),4,&rgba);gpu.transfer(mv.Get(),2,&motion);
        phoenix_cuda_d3d12_frame frame{};frame.struct_bytes=sizeof(frame);frame.version=1;frame.queue=gpu.queue.Get();
        frame.color=color.Get();frame.motion=mv.Get();frame.output=output.Get();frame.motion_scale_x=frame.motion_scale_y=1;
        auto stats=[](std::vector<double> values){std::sort(values.begin(),values.end());const size_t n=values.size();return Json{{"median_ms",(values[(n-1)/2]+values[n/2])*.5},{"min_ms",values.front()},{"max_ms",values.back()}};};
        auto sample=[&]{
            const std::string text="{\"reset\":true}";frame.metadata_json=text.c_str();
            for(unsigned i=0;i<20;++i)demand(evaluate(session.get(),&frame)==0,error());
            std::vector<double> wall,network;
            for(unsigned i=0;i<30;++i){const auto start=std::chrono::steady_clock::now();demand(evaluate(session.get(),&frame)==0,error());
                wall.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());network.push_back(state().at("operator_ms").get<double>());}
            const auto result=gpu.transfer(output.Get(),4);
            return Json{{"wall",stats(wall)},{"gpu",stats(network)},{"samples_ms",wall},{"gpu_samples_ms",network},{"status",state()},
                {"output_sha256",phoenix_model::weight_digest(reinterpret_cast<const uint8_t*>(result.data()),result.size()*2)}};
        };
        auto fresh=sample();
        std::string metadata="{\"reset\":true,\"repeats\":[1,0,2,1,0,1,1,1]}";frame.metadata_json=metadata.c_str();demand(evaluate(session.get(),&frame)==0,error());
        auto restored=sample();demand(fresh.at("output_sha256")==restored.at("output_sha256"),"Restoring full graph changed output");
        gpu.check_debug();session.reset();FreeLibrary(module);
        std::ofstream{fs::path(argv[2])}<<Json{{"passed",true},{"fresh",fresh},{"restored",restored}}.dump(2)<<'\n';
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
