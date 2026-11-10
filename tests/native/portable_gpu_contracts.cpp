// Explicit hardware contract, never registered as an automatic GPU test.
#include "../../src/integration/compatibility/portable_gpu.h"
#include "../../src/integration/compatibility/portable_policy.h"
#include "../../native_runtime/phoenix.h"
#include <chrono>
#include <algorithm>
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

int wmain(int argc,wchar_t** argv){try{
 demand(argc==3,"Usage: transport-contract job.json report.json");Json job;{std::ifstream f{fs::path(argv[1])};f>>job;}
 const auto root=fs::u8path(job.at("runtime_root").get<std::string>());const unsigned w=job.at("width"),h=job.at("height");const size_t pixels=size_t(w)*h;
 auto runner=job.at("config");const auto producer=job.value("producer_device",runner.at("device").get<std::string>());
 demand(cuInit(0)==CUDA_SUCCESS,"CUDA initialization failed");CUdevice selected=-1;int count{};cuDeviceGetCount(&count);
 for(int i=0;i<count;++i){CUuuid uuid{};cuDeviceGetUuid(&uuid,i);std::string id="cuda:";for(unsigned char b:uuid.bytes){char hex[3];sprintf_s(hex,"%02x",b);id+=hex;}if(id==producer)selected=i;}
 demand(selected>=0,"CUDA UUID not present");Gpu gpu(selected);
 ComPtr<IDXGIFactory4> factory;hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));ComPtr<IDXGIAdapter> adapter;hr(factory->EnumAdapterByLuid(gpu.device->GetAdapterLuid(),IID_PPV_ARGS(&adapter)));
 ComPtr<ID3D11Device> game11;ComPtr<ID3D11DeviceContext> context11;D3D_FEATURE_LEVEL level=D3D_FEATURE_LEVEL_11_0;
 hr(D3D11CreateDevice(adapter.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,&level,1,D3D11_SDK_VERSION,&game11,nullptr,&context11));
 demand(game11->GetFeatureLevel()==D3D_FEATURE_LEVEL_11_0,"Test must exercise a DX11 feature-level-11.0 device");
 HANDLE input_handle{},output_handle{};
 auto color11=phoenix_compat::portable_gpu_texture11(game11.Get(),w,h,&input_handle),output11=phoenix_compat::portable_gpu_texture11(game11.Get(),w,h,&output_handle);
 ComPtr<ID3D12Resource> color,output;hr(gpu.device->OpenSharedHandle(input_handle,IID_PPV_ARGS(&color)));hr(gpu.device->OpenSharedHandle(output_handle,IID_PPV_ARGS(&output)));
 phoenix_compat::PortableGpuHandles handles(input_handle,output_handle,gpu.device->GetAdapterLuid());CloseHandle(input_handle);CloseHandle(output_handle);
 phoenix_compat::PortableGpuFrames worker(handles,w,h);
 phoenix_cpp::PipelineClient cpu(root,gpu.device.Get(),gpu.queue.Get()),shared(root,worker.device(),worker.queue());
 std::vector<float> source(pixels*3);{std::ifstream f(fs::u8path(job.at("input").get<std::string>()),std::ios::binary);f.read(reinterpret_cast<char*>(source.data()),source.size()*4);demand(bool(f),"Input cannot be read");}
 std::vector<uint16_t> bytes(pixels*2);auto* rgba=reinterpret_cast<uint8_t*>(bytes.data());
 for(size_t p=0;p<pixels;++p){for(unsigned c=0;c<3;++c){rgba[p*4+c]=uint8_t(std::clamp(source[p*3+c],0.f,1.f)*255.f+.5f);source[p*3+c]=rgba[p*4+c]/255.f;}rgba[p*4+3]=uint8_t(p%256);}
 gpu.transfer(color.Get(),2,&bytes);
 phoenix_compat::Settings settings;settings.mode=phoenix_addon::phoenix::ours;settings.pipeline.motion_enabled=true;settings.pipeline.scene_detection=true;settings.pipeline.monitoring=1;
 demand(phoenix_compat::portable_gpu_settings(settings,w,h),"Temporal GPU settings declined");
 auto excluded=settings;excluded.mode=phoenix_addon::phoenix::split;demand(!phoenix_compat::portable_gpu_settings(excluded,w,h),"Comparison view admitted");
 excluded=settings;excluded.processing.enabled=true;excluded.processing.scale=50;demand(!phoenix_compat::portable_gpu_settings(excluded,w,h),"Nonidentity CPU processing admitted");
 Json times={{"host_ms",Json::array()},{"gpu_ms",Json::array()}},frame_hashes=Json::array();unsigned frames=0;
 for(unsigned i=0;i<28;++i){
  auto metadata=phoenix_compat::portable_metadata(settings,i==0,true);
  if(i>=20)metadata["controls"]={{"local_tone",i>=24?2.f:0.f},{"local_structure",i>=24?2.f:0.f},{"auto_mask",true},{"intensity",i==22?.5f:1.f},{"style",i==23?2:0}};
  const auto begin=std::chrono::steady_clock::now();auto expected=cpu.frame(root,runner,source,w,h,w,h,metadata,i+1,0,0);const auto mid=std::chrono::steady_clock::now();
  worker.frame(shared,root,runner,w,h,metadata,i+1,0,0,{});const auto end=std::chrono::steady_clock::now();
  const auto words=gpu.transfer(output.Get(),2);const auto* actual=reinterpret_cast<const uint8_t*>(words.data());
  frame_hashes.push_back(phoenix_model::weight_digest(reinterpret_cast<const uint8_t*>(expected.data()),expected.size()*sizeof(float)));
  for(size_t p=0;p<pixels;++p){for(unsigned c=0;c<3;++c)demand(actual[p*4+c]==phoenix_compat::cpu_color::unorm8(expected[p*3+c]),"GPU mailbox differs from host codec/pipeline");demand(actual[p*4+3]==rgba[p*4+3],"GPU mailbox changed source alpha");}
  if(i>=5&&i<20){times["host_ms"].push_back(std::chrono::duration<double,std::milli>(mid-begin).count());times["gpu_ms"].push_back(std::chrono::duration<double,std::milli>(end-mid).count());}++frames;
 }
 // A requested pair is the only per-frame pixel readback in this worker.
 demand(worker.request_pair(fs::path(argv[2]).parent_path()/"pair"),"Pair request rejected");auto metadata=phoenix_compat::portable_metadata(settings,true,true);
 worker.frame(shared,root,runner,w,h,metadata,29,0,0,{});demand(worker.pair_status().at("state")=="saved","GPU screenshot pair not saved");gpu.check_debug();
 ComPtr<ID3D12InfoQueue> messages;if(SUCCEEDED(worker.device()->QueryInterface(IID_PPV_ARGS(&messages)))){
  for(uint64_t i=0;i<messages->GetNumStoredMessagesAllowedByRetrievalFilter();++i){SIZE_T size{};messages->GetMessage(i,nullptr,&size);std::vector<uint8_t> bytes(size);auto* message=reinterpret_cast<D3D12_MESSAGE*>(bytes.data());hr(messages->GetMessage(i,message,&size));demand(message->Severity>D3D12_MESSAGE_SEVERITY_ERROR,message->pDescription);}}
 std::ofstream report{fs::path(argv[2])};report<<Json{{"passed",true},{"exact_frames",frames},{"frame_hashes",frame_hashes},{"alpha_exact",true},{"debug_layer",bool(gpu.messages)},{"times",times},{"status",shared.status()},{"pair",worker.pair_status()},{"policy_rejections",{"comparison","nonidentity processing"}}}.dump(2)<<'\n';
 std::cout<<"GPU mailbox: "<<frames<<" frames exact, alpha and on-demand pair passed\n";return 0;
 }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
