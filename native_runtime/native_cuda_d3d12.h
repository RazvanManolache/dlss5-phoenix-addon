#pragma once
#ifdef _WIN32
#include "phoenix_cuda_d3d12.h"
#include <wrl/client.h>
#include <memory>

namespace phoenix_native_cuda {
using Microsoft::WRL::ComPtr;
inline void d3d_check(HRESULT value){if(FAILED(value))throw std::runtime_error("Native CUDA D3D12 failure: "+std::to_string(uint32_t(value)));}
struct InteropUnavailable:std::runtime_error{using std::runtime_error::runtime_error;};
struct SharedHandle {
    HANDLE value{};
    ~SharedHandle(){if(value)CloseHandle(value);}
};
struct SharedSemaphore {
    cudaExternalSemaphore_t value{};
    ~SharedSemaphore(){if(value)cudaDestroyExternalSemaphore(value);}
};
template<class T>ComPtr<T> unwrapped(T* source){
    constexpr GUID iid={0x7f2c9a11,0x3b4e,0x4d6a,{0x81,0x2f,0x5e,0x9c,0xd3,0x7a,0x1b,0x42}};
    ComPtr<T> result=source;ComPtr<IUnknown> native;
    if(SUCCEEDED(source->QueryInterface(iid,reinterpret_cast<void**>(native.GetAddressOf()))))d3d_check(native.As(&result));
    return result;
}
struct SharedImage {
    ComPtr<ID3D12Resource> resource;
    cudaExternalMemory_t memory{};
    cudaMipmappedArray_t mipmap{};
    cudaArray_t array{};
    cudaTextureObject_t texture{};
    cudaSurfaceObject_t surface{};
    uint64_t bytes{};
    ~SharedImage(){
        if(texture)cudaDestroyTextureObject(texture);
        if(surface)cudaDestroySurfaceObject(surface);
        if(mipmap)cudaFreeMipmappedArray(mipmap);
        if(memory)cudaDestroyExternalMemory(memory);
    }
    void initialize(ID3D12Device* device,unsigned width,unsigned height,bool motion){
        D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width=width;desc.Height=height;desc.DepthOrArraySize=1;desc.MipLevels=1;
        desc.Format=motion?DXGI_FORMAT_R16G16_FLOAT:DXGI_FORMAT_R16G16B16A16_FLOAT;
        desc.SampleDesc.Count=1;desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;heap.CreationNodeMask=heap.VisibleNodeMask=1;
        d3d_check(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_SHARED,&desc,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&resource)));
        bytes=device->GetResourceAllocationInfo(0,1,&desc).SizeInBytes;
        SharedHandle handle;d3d_check(device->CreateSharedHandle(resource.Get(),nullptr,GENERIC_ALL,nullptr,&handle.value));
        cudaExternalMemoryHandleDesc external{};external.type=cudaExternalMemoryHandleTypeD3D12Resource;
        external.handle.win32.handle=handle.value;external.size=bytes;external.flags=cudaExternalMemoryDedicated;
        check(cudaImportExternalMemory(&memory,&external));
        cudaExternalMemoryMipmappedArrayDesc mapped{};
        mapped.formatDesc={16,16,motion?0:16,motion?0:16,cudaChannelFormatKindFloat};
        mapped.extent=make_cudaExtent(width,height,0);mapped.flags=cudaArraySurfaceLoadStore;mapped.numLevels=1;
        check(cudaExternalMemoryGetMappedMipmappedArray(&mipmap,memory,&mapped));
        check(cudaGetMipmappedArrayLevel(&array,mipmap,0));
        cudaResourceDesc view{};view.resType=cudaResourceTypeArray;view.res.array.array=array;
        cudaTextureDesc sampler{};sampler.addressMode[0]=sampler.addressMode[1]=cudaAddressModeClamp;
        sampler.filterMode=cudaFilterModePoint;sampler.readMode=cudaReadModeElementType;sampler.normalizedCoords=1;
        check(cudaCreateTextureObject(&texture,&view,&sampler,nullptr));
        check(cudaCreateSurfaceObject(&surface,&view));
    }
};
class D3D12Bridge {
    PrimaryContext context_;
    ComPtr<ID3D12CommandQueue> queue_lease_,queue_;
    ComPtr<ID3D12Device> device_lease_,device_;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> command_;
    ComPtr<ID3D12Fence> fence_;
    SharedHandle event_;
    SharedSemaphore semaphore_;
    uint64_t serial_{},retirement_{};
    cudaStream_t stream_{};
    unsigned width_,height_;
    SharedImage input_,motion_,output_;
    static void transition(ID3D12GraphicsCommandList* command,ID3D12Resource* image,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after){
        if(before==after)return;
        D3D12_RESOURCE_BARRIER barrier{};barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition={image,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after};command->ResourceBarrier(1,&barrier);
    }
    void wait(uint64_t value){
        const auto complete=fence_->GetCompletedValue();
        if(complete==UINT64_MAX)throw std::runtime_error("D3D12 interop device lost");
        if(complete<value){d3d_check(fence_->SetEventOnCompletion(value,event_.value));
            if(WaitForSingleObject(event_.value,30000)!=WAIT_OBJECT_0)throw std::runtime_error("D3D12 interop completion timed out");}
        d3d_check(device_->GetDeviceRemovedReason());
    }
    void begin(){wait(retirement_);d3d_check(allocator_->Reset());d3d_check(command_->Reset(allocator_.Get(),nullptr));}
    uint64_t submit(){
        d3d_check(command_->Close());ID3D12CommandList* commands[]{command_.Get()};queue_->ExecuteCommandLists(1,commands);
        retirement_=++serial_;d3d_check(queue_->Signal(fence_.Get(),retirement_));return retirement_;
    }
    void validate_image(ID3D12Resource* image,DXGI_FORMAT format){
        ComPtr<ID3D12Device> owner;d3d_check(image->GetDevice(IID_PPV_ARGS(&owner)));owner=unwrapped(owner.Get());
        ComPtr<IUnknown> one,two;d3d_check(owner.As(&one));d3d_check(device_.As(&two));
        if(one.Get()!=two.Get())throw std::runtime_error("D3D12 interop texture belongs to a different device");
        const auto desc=image->GetDesc();
        if(desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||desc.Width!=width_||desc.Height!=height_||
           desc.Format!=format||desc.SampleDesc.Count!=1||desc.DepthOrArraySize!=1||desc.MipLevels!=1)
            throw std::runtime_error("D3D12 interop texture format or extent mismatch");
    }
public:
    D3D12Bridge(int ordinal,unsigned width,unsigned height,ID3D12CommandQueue* queue):width_(width),height_(height){
        context_.retain(ordinal);queue_lease_=queue;queue_=unwrapped(queue);
        d3d_check(queue_->GetDevice(IID_PPV_ARGS(&device_lease_)));device_=unwrapped(device_lease_.Get());
        cudaDeviceProp gpu{};check(cudaGetDeviceProperties(&gpu,ordinal));const auto luid=device_->GetAdapterLuid();
        if(device_->GetNodeCount()!=1||std::memcmp(gpu.luid,&luid,sizeof(luid))!=0)
            throw InteropUnavailable("D3D12 device differs from the selected CUDA GPU");
        if(queue_->GetDesc().Type!=D3D12_COMMAND_LIST_TYPE_DIRECT)throw std::runtime_error("D3D12 interop requires a DIRECT queue");
        d3d_check(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator_)));
        d3d_check(device_->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator_.Get(),nullptr,IID_PPV_ARGS(&command_)));d3d_check(command_->Close());
        d3d_check(device_->CreateFence(0,D3D12_FENCE_FLAG_SHARED,IID_PPV_ARGS(&fence_)));
        event_.value=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!event_.value)throw std::runtime_error("Cannot create D3D12 interop event");
        SharedHandle handle;d3d_check(device_->CreateSharedHandle(fence_.Get(),nullptr,GENERIC_ALL,nullptr,&handle.value));
        cudaExternalSemaphoreHandleDesc desc{};desc.type=cudaExternalSemaphoreHandleTypeD3D12Fence;desc.handle.win32.handle=handle.value;
        check(cudaImportExternalSemaphore(&semaphore_.value,&desc));
        input_.initialize(device_.Get(),width,height,false);output_.initialize(device_.Get(),width,height,false);
    }
    ~D3D12Bridge(){
        if(context_.value)(void)cuCtxSetCurrent(context_.value);
        if(stream_)(void)cudaStreamSynchronize(stream_);
    }
    void validate(const phoenix_cuda_d3d12_frame& frame){
        if(unwrapped(frame.queue).Get()!=queue_.Get())throw std::runtime_error("D3D12 interop queue changed; recreate session");
        validate_image(frame.color,DXGI_FORMAT_R16G16B16A16_FLOAT);validate_image(frame.output,DXGI_FORMAT_R16G16B16A16_FLOAT);
        if(frame.motion)validate_image(frame.motion,DXGI_FORMAT_R16G16_FLOAT);
    }
    void prepare(const phoenix_cuda_d3d12_frame& frame,cudaStream_t stream){
        context_.activate();stream_=stream;
        if(frame.motion&&!motion_.resource)motion_.initialize(device_.Get(),width_,height_,true);
        begin();
        auto copy=[&](ID3D12Resource* source,D3D12_RESOURCE_STATES state,ID3D12Resource* destination){
            transition(command_.Get(),source,state,D3D12_RESOURCE_STATE_COPY_SOURCE);
            transition(command_.Get(),destination,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
            command_->CopyResource(destination,source);
            transition(command_.Get(),destination,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);
            transition(command_.Get(),source,D3D12_RESOURCE_STATE_COPY_SOURCE,state);
        };
        copy(frame.color,frame.color_state,input_.resource.Get());
        if(frame.motion)copy(frame.motion,frame.motion_state,motion_.resource.Get());
        cudaExternalSemaphoreWaitParams wait{};wait.params.fence.value=submit();
        check(cudaWaitExternalSemaphoresAsync(&semaphore_.value,&wait,1,stream));
    }
    uint64_t signal_cuda(cudaStream_t stream){
        cudaExternalSemaphoreSignalParams signal{};signal.params.fence.value=++serial_;
        check(cudaSignalExternalSemaphoresAsync(&semaphore_.value,&signal,1,stream));
        return signal.params.fence.value;
    }
    void finish_completed(const phoenix_cuda_d3d12_frame& frame,uint64_t value){
        // The executor's validation readback has already completed this signal
        // and resolved CUDA errors. Never enqueue a D3D12 wait on pending CUDA.
        if(!value||value!=serial_)throw std::runtime_error("Invalid completed CUDA signal");
        d3d_check(queue_->Wait(fence_.Get(),value));begin();
        transition(command_.Get(),output_.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(command_.Get(),frame.output,frame.output_state,D3D12_RESOURCE_STATE_COPY_DEST);
        command_->CopyResource(frame.output,output_.resource.Get());
        transition(command_.Get(),frame.output,D3D12_RESOURCE_STATE_COPY_DEST,frame.output_state);
        transition(command_.Get(),output_.resource.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
        wait(submit());
    }
    cudaTextureObject_t color()const{return input_.texture;}
    cudaTextureObject_t motion()const{return motion_.texture;}
    cudaSurfaceObject_t alpha()const{return input_.surface;}
    cudaSurfaceObject_t output()const{return output_.surface;}
    cudaArray_t output_array()const{return output_.array;}
    uint64_t bytes()const{return input_.bytes+motion_.bytes+output_.bytes;}
};
}
#endif
