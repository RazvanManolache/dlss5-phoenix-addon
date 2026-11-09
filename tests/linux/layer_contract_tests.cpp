// Fake downstream Vulkan dispatch: no loader, physical device, game or model.
// Check binary semaphore consumption and presentation preservation under faults.
#include "../../src/integration/linux/layer.cpp"
#include <array>

namespace {
int presents{}, submits{}, inference{}, resetCalls{}, failSubmit{}, failReset{};
bool inferenceFails{}, appWaitAvailable{}, layerSignalAvailable{};
VkSemaphore appWait=reinterpret_cast<VkSemaphore>(uintptr_t(11));
VkSemaphore layerSignal=reinterpret_cast<VkSemaphore>(uintptr_t(12));
std::array<uint8_t,4> imageBytes{}, mappedBytes{};
const void* expectedChain{};
VkResult* expectedResults{};
void require(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr,"Contract failed: %s\n",message); std::exit(1); }
}
VKAPI_ATTR VkResult VKAPI_CALL fakeReset(VkCommandBuffer, VkCommandBufferResetFlags) {
    return ++resetCalls==failReset ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL fakeBegin(VkCommandBuffer,const VkCommandBufferBeginInfo*) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL fakeEnd(VkCommandBuffer) { return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL fakeBarrier(VkCommandBuffer,VkPipelineStageFlags,VkPipelineStageFlags,VkDependencyFlags,
    uint32_t,const VkMemoryBarrier*,uint32_t,const VkBufferMemoryBarrier*,uint32_t,const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL fakeRead(VkCommandBuffer,VkImage,VkImageLayout,VkBuffer,uint32_t,const VkBufferImageCopy*) { mappedBytes=imageBytes; }
VKAPI_ATTR void VKAPI_CALL fakeWrite(VkCommandBuffer,VkBuffer,VkImage,VkImageLayout,uint32_t,const VkBufferImageCopy*) { imageBytes=mappedBytes; }
VKAPI_ATTR VkResult VKAPI_CALL fakeFence(VkDevice,uint32_t,const VkFence*) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL fakeWait(VkDevice,uint32_t,const VkFence*,VkBool32,uint64_t) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL fakeSubmit(VkQueue,uint32_t count,const VkSubmitInfo* info,VkFence) {
    require(count==1,"one submission");
    ++submits;
    if (submits==failSubmit) return VK_ERROR_DEVICE_LOST;
    for (uint32_t i=0;i<info->waitSemaphoreCount;++i) {
        require(info->pWaitSemaphores[i]==appWait && appWaitAvailable,"app wait consumed exactly once");
        require(info->pWaitDstStageMask[i]==VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,"wait covers image operations");
        appWaitAvailable=false;
    }
    if (info->signalSemaphoreCount) {
        require(info->pSignalSemaphores[0]==layerSignal && !layerSignalAvailable,"per-image completion signalled once");
        layerSignalAvailable=true;
    }
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL fakePresent(VkQueue,const VkPresentInfoKHR* info) {
    ++presents;
    require(info->pNext==expectedChain,"presentation extension chain preserved");
    require(info->pResults==expectedResults,"per-swapchain results pointer preserved");
    require(info->waitSemaphoreCount==1,"presentation wait count");
    if (info->pWaitSemaphores[0]==appWait) { require(appWaitAvailable,"original semaphore not consumed twice"); appWaitAvailable=false; }
    else { require(info->pWaitSemaphores[0]==layerSignal && layerSignalAvailable,"wait on layer completion"); layerSignalAvailable=false; }
    return VK_SUBOPTIMAL_KHR;
}
int fakeFrame(void*,const phoenix_frame_v2* frame) {
    ++inference;
    require(runtimeCallDepth>0,"inference keeps runtime-owned instance creation separate from game visibility");
    require(frame->version==2 && frame->struct_bytes==sizeof(*frame),"bounded frame interface");
    require(frame->color.capacity_bytes==12 && frame->output.capacity_bytes==12,"full-extent bounded RGB");
    require(!frame->motion.data && !frame->depth.data && !frame->validity.data,"no invented guides");
    require(!std::strcmp(frame->metadata_json,"{\"reset\":true}"),"no temporal reuse without guides");
    if (inferenceFails) return -1;
    frame->output.data[0]=1.f; frame->output.data[1]=0.f; frame->output.data[2]=0.f;
    return 0;
}
int fakeRgba8(void*, const phoenix_rgba8_frame* frame) {
    ++inference;
    require(runtimeCallDepth>0,"packed inference scopes runtime-owned instances");
    require(frame->struct_bytes==sizeof(*frame) && frame->version==1 &&
        frame->width==1 && frame->height==1 && frame->capacity_bytes==4 &&
        frame->row_stride_bytes==0 && frame->format==0 && frame->flags==1,
        "bounded packed reset frame interface");
    auto* bytes=static_cast<uint8_t*>(frame->data);
    bytes[0]=255; bytes[1]=0; bytes[2]=0;
    return 0;
}
const char* fakeError() { return "injected inference failure"; }
void resetScenario() {
    presents=submits=inference=resetCalls=failSubmit=failReset=0;
    inferenceFails=false; appWaitAvailable=true; layerSignalAvailable=false;
    imageBytes={20,40,60,77}; mappedBytes={}; expectedChain=nullptr; expectedResults=nullptr;
}
}

int main() {
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryTypeCount=4;
    memory.memoryTypes[0].propertyFlags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    memory.memoryTypes[1].propertyFlags=memory.memoryTypes[0].propertyFlags | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    memory.memoryTypes[2].propertyFlags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    memory.memoryTypes[3].propertyFlags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    require(transferMemoryType(memory,15)==1,"cached coherent readback preferred over first coherent type");
    require(transferMemoryType(memory,13)==0,"coherent fallback when cached coherent memory is excluded");
    require(transferMemoryType(memory,12)==UINT32_MAX,"noncoherent and device-only memory rejected");
    require(transferMemoryType(memory,0)==UINT32_MAX,"memory requirement mask respected");

    require(vkGetInstanceProcAddr(nullptr,"vkQueuePresentKHR")==nullptr,"device commands unavailable without an instance");
    require(vkGetInstanceProcAddr(nullptr,"vkCreateInstance")!=nullptr,"global instance creation available");
    VkNegotiateLayerInterface negotiation{}; negotiation.sType=LAYER_NEGOTIATE_INTERFACE_STRUCT;
    negotiation.loaderLayerInterfaceVersion=1;
    require(vkNegotiateLoaderLayerInterfaceVersion(&negotiation)==VK_ERROR_INITIALIZATION_FAILED,"reject obsolete loader interface");
    negotiation.loaderLayerInterfaceVersion=5;
    require(vkNegotiateLoaderLayerInterfaceVersion(&negotiation)==VK_SUCCESS && negotiation.loaderLayerInterfaceVersion==2,"negotiate loader interface");
    int dispatch{}; void* dispatchKey=&dispatch;
    VkDevice device=reinterpret_cast<VkDevice>(&dispatchKey);
    VkQueue queue=reinterpret_cast<VkQueue>(&dispatchKey);
    auto d=std::make_shared<Device>(); d->handle=device;
    d->ResetCommandBuffer=fakeReset; d->BeginCommandBuffer=fakeBegin; d->EndCommandBuffer=fakeEnd;
    d->CmdPipelineBarrier=fakeBarrier; d->CmdCopyImageToBuffer=fakeRead; d->CmdCopyBufferToImage=fakeWrite;
    d->ResetFences=fakeFence; d->WaitForFences=fakeWait; d->QueueSubmit=fakeSubmit; d->QueuePresentKHR=fakePresent;
    d->frame=fakeFrame; d->error=fakeError; d->queues[queue]={0,true};
    VkSwapchainKHR swap=reinterpret_cast<VkSwapchainKHR>(uintptr_t(1));
    auto s=std::make_unique<Swapchain>(); s->handle=swap; s->extent={1,1}; s->eligible=s->ready=true;
    s->family=0; s->diagnostics=true; s->images.push_back(reinterpret_cast<VkImage>(uintptr_t(2)));
    s->complete.push_back(layerSignal); s->mapped=mappedBytes.data(); s->input.resize(3); s->output.resize(3);
    d->swaps[swap]=std::move(s); devices[key(device)]=d;
    uint32_t index=0;
    VkPresentInfoKHR info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR}; info.swapchainCount=1; info.pSwapchains=&swap;
    info.pImageIndices=&index; info.waitSemaphoreCount=1; info.pWaitSemaphores=&appWait;
    setenv("PHOENIX_ENABLE","1",1);
    resetScenario();
    VkResult result{}; info.pResults=&result; expectedResults=&result;
    require(vkQueuePresentKHR(queue,&info)==VK_SUBOPTIMAL_KHR,"preserve downstream result");
    require(presents==1 && submits==2 && inference==1,"capture, inference, upload, presentation");
    require(d->swaps[swap]->processed==1 && d->swaps[swap]->presented==1 && d->swaps[swap]->changed==1,"diagnostics count completed, changed and presented frames");
    require(imageBytes==std::array<uint8_t,4>{255,0,0,77},"processed RGB with original alpha");
    info.pResults=nullptr;

    resetScenario(); d->frameRgba8=fakeRgba8;
    d->swaps[swap]->input.clear(); d->swaps[swap]->output.clear();
    require(vkQueuePresentKHR(queue,&info)==VK_SUBOPTIMAL_KHR && inference==1,
        "optional packed runtime path executes without host float planes");
    require(imageBytes==std::array<uint8_t,4>{255,0,0,77},"packed output and alpha preserved");
    d->frameRgba8=nullptr;
    d->swaps[swap]->input.resize(3); d->swaps[swap]->output.resize(3);

    resetScenario(); inferenceFails=true;
    require(vkQueuePresentKHR(queue,&info)==VK_SUBOPTIMAL_KHR,"inference failure presents original");
    require(imageBytes==std::array<uint8_t,4>{20,40,60,77},"original pixels retained");
    require(d->swaps[swap]->failed==1,"diagnostics count inference failures");

    resetScenario(); failReset=1;
    require(vkQueuePresentKHR(queue,&info)==VK_SUBOPTIMAL_KHR && submits==0 && inference==0,"record failure before consuming app wait passes through");
    resetScenario(); failSubmit=1;
    require(vkQueuePresentKHR(queue,&info)==VK_ERROR_DEVICE_LOST && presents==0 && inference==0,"capture submit error does not reuse waits");
    resetScenario(); failSubmit=2;
    require(vkQueuePresentKHR(queue,&info)==VK_ERROR_DEVICE_LOST && presents==0 && !appWaitAvailable,"upload failure never forwards consumed app wait");
    resetScenario(); failReset=2;
    require(vkQueuePresentKHR(queue,&info)==VK_ERROR_OUT_OF_DEVICE_MEMORY && presents==0,"upload recording failure propagates after app waits consumed");

    resetScenario(); VkBaseInStructure chain{}; info.pNext=&chain; expectedChain=&chain;
    require(vkQueuePresentKHR(queue,&info)==VK_SUBOPTIMAL_KHR && submits==0,"extension path passes through untouched");
    info.pNext=nullptr;
    resetScenario(); d->queues[queue].graphics=false;
    require(vkQueuePresentKHR(queue,&info)==VK_SUBOPTIMAL_KHR && submits==0,"non-graphics queue passes through");
    d->queues[queue].graphics=true;
    resetScenario(); setenv("PHOENIX_ENABLE","0",1);
    require(vkQueuePresentKHR(queue,&info)==VK_SUBOPTIMAL_KHR && submits==0,"opt-in required");
    devices.clear();
}
