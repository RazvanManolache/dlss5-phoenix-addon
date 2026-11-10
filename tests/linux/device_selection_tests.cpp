// Independent Vulkan-instance visibility using fake downstream dispatch only.
#include "../../src/integration/linux/layer.cpp"
#include <thread>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr,"Device selection contract failed: %s\n",message); std::exit(1); }
}
int dispatchTokens[8]{};
void* instanceHandles[8]{};
std::atomic<unsigned> created{};
int amdHandle{},nvidiaHandle{};
VkPhysicalDevice amd=reinterpret_cast<VkPhysicalDevice>(&amdHandle);
VkPhysicalDevice nvidia=reinterpret_cast<VkPhysicalDevice>(&nvidiaHandle);
const char* amdID="vulkan:0123456789abcdef0123456789abcdef";
std::atomic<bool> incompleteOnce{};
VKAPI_ATTR VkResult VKAPI_CALL create(const VkInstanceCreateInfo*,const VkAllocationCallbacks*,VkInstance* output) {
    unsigned index=created++;
    require(index<8,"bounded fake instances");
    instanceHandles[index]=&dispatchTokens[index];
    *output=reinterpret_cast<VkInstance>(&instanceHandles[index]);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy(VkInstance,const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL enumerate(VkInstance,uint32_t* count,VkPhysicalDevice* values) {
    if (!values) { *count=incompleteOnce ? 1 : 2; return VK_SUCCESS; }
    if (incompleteOnce.exchange(false)) { values[0]=nvidia; *count=1; return VK_INCOMPLETE; }
    const uint32_t available=*count;
    if (available) values[0]=nvidia;
    if (available>1) values[1]=amd;
    *count=std::min(available,2u); return available<2 ? VK_INCOMPLETE : VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL properties(VkPhysicalDevice device,VkPhysicalDeviceProperties2* output) {
    auto* id=static_cast<VkPhysicalDeviceIDProperties*>(output->pNext);
    std::array<uint8_t,VK_UUID_SIZE> uuid{};
    require(parseDevice(device==amd ? amdID : "vulkan:fedcba9876543210fedcba9876543210",uuid),"fixture UUID");
    std::memcpy(id->deviceUUID,uuid.data(),uuid.size());
    std::strcpy(output->properties.deviceName,device==amd ? "Display AMD" : "Compute NVIDIA");
}
VKAPI_ATTR VkResult VKAPI_CALL groups(VkInstance,uint32_t* count,VkPhysicalDeviceGroupProperties* values) {
    if (!values) { *count=1; return VK_SUCCESS; }
    if (!*count) return VK_INCOMPLETE;
    values[0].physicalDeviceCount=2; values[0].physicalDevices[0]=nvidia; values[0].physicalDevices[1]=amd;
    *count=1; return VK_SUCCESS;
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL gpa(VkInstance,const char* name) {
    if (!std::strcmp(name,"vkCreateInstance")) return reinterpret_cast<PFN_vkVoidFunction>(create);
    if (!std::strcmp(name,"vkDestroyInstance")) return reinterpret_cast<PFN_vkVoidFunction>(destroy);
    if (!std::strcmp(name,"vkEnumeratePhysicalDevices")) return reinterpret_cast<PFN_vkVoidFunction>(enumerate);
    if (!std::strcmp(name,"vkGetPhysicalDeviceProperties2")) return reinterpret_cast<PFN_vkVoidFunction>(properties);
    if (!std::strcmp(name,"vkEnumeratePhysicalDeviceGroups") || !std::strcmp(name,"vkEnumeratePhysicalDeviceGroupsKHR"))
        return reinterpret_cast<PFN_vkVoidFunction>(groups);
    return nullptr;
}
VkResult make(VkInstance& output, uint32_t apiVersion=VK_API_VERSION_1_1) {
    VkLayerInstanceLink link{}; link.pfnNextGetInstanceProcAddr=gpa;
    VkLayerInstanceCreateInfo chain{}; chain.sType=VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO;
    chain.function=VK_LAYER_LINK_INFO; chain.u.pLayerInfo=&link;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=apiVersion;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; info.pNext=&chain; info.pApplicationInfo=&app;
    return vkCreateInstance(&info,nullptr,&output);
}
}
int main() {
    std::array<uint8_t,VK_UUID_SIZE> uuid{};
    require(parseDevice(amdID,uuid),"parse UUID");
    require(!parseDevice("vulkan:00112233445566778899aabbccddee  ",uuid),"reject whitespace");
    setenv("PHOENIX_GAME_DEVICE",amdID,1);
    VkInstance obsolete{};
    require(make(obsolete,VK_API_VERSION_1_0)==VK_ERROR_INCOMPATIBLE_DRIVER && created==0,"selection rejects instances without core UUID queries");
    VkInstance game{},runtime{},otherThread{};
    require(make(game)==VK_SUCCESS,"create filtered game instance");
    {
        RuntimeCall call;
        require(make(runtime)==VK_SUCCESS,"create runtime instance");
        std::thread thread([&] { require(make(otherThread)==VK_SUCCESS,"create independent game-thread instance"); });
        thread.join();
    }
    require(runtimeCallDepth==0,"runtime scope unwinds");
    require(instanceFor(game)->filterDevices && !instanceFor(game)->runtimeOwned,"game instance filtered");
    require(!instanceFor(runtime)->filterDevices && instanceFor(runtime)->runtimeOwned,"runtime instance unfiltered after scope ends");
    require(instanceFor(otherThread)->filterDevices && !instanceFor(otherThread)->runtimeOwned,"runtime scope is thread local");
    require(!std::strcmp(std::getenv("PHOENIX_GAME_DEVICE"),amdID),"runtime never changes process environment");
    uint32_t count=0;
    require(vkEnumeratePhysicalDevices(game,&count,nullptr)==VK_SUCCESS && count==1,"game sees one GPU");
    VkPhysicalDevice output[2]{}; count=2;
    require(vkEnumeratePhysicalDevices(game,&count,output)==VK_SUCCESS && count==1 && output[0]==amd,"UUID selection is independent of driver order");
    count=0; output[0]=nvidia;
    require(vkEnumeratePhysicalDevices(game,&count,output)==VK_INCOMPLETE && count==0 && output[0]==nvidia,"zero-capacity enumeration does not write");
    incompleteOnce=true; count=2;
    require(vkEnumeratePhysicalDevices(game,&count,output)==VK_SUCCESS && output[0]==amd,"retry changing downstream device count");
    count=2;
    require(vkEnumeratePhysicalDevices(runtime,&count,output)==VK_SUCCESS && count==2 && output[0]==nvidia && output[1]==amd,"runtime retains inference GPU");
    VkPhysicalDeviceGroupProperties group{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES};
    int extension{}; group.pNext=&extension; count=1;
    require(vkEnumeratePhysicalDeviceGroups(game,&count,&group)==VK_SUCCESS && group.physicalDeviceCount==1 && group.physicalDevices[0]==amd,"group enumeration hides other GPUs too");
    require(group.pNext==&extension && group.sType==VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES,"group output preserves chain");
    count=1;
    require(vkEnumeratePhysicalDeviceGroupsKHR(runtime,&count,&group)==VK_SUCCESS && group.physicalDeviceCount==2,"runtime KHR group enumeration remains intact");
    setenv("PHOENIX_GAME_DEVICE","vulkan:ffffffffffffffffffffffffffffffff",1);
    VkInstance missing{}; require(make(missing)==VK_SUCCESS,"create instance with unavailable UUID"); count=2;
    require(vkEnumeratePhysicalDevices(missing,&count,output)==VK_ERROR_INITIALIZATION_FAILED && count==0,"unavailable UUID never silently substitutes a GPU");
    setenv("PHOENIX_GAME_DEVICE","invalid",1);
    VkInstance invalid{}; unsigned before=created;
    require(make(invalid)==VK_ERROR_INITIALIZATION_FAILED && created==before,"malformed UUID rejected before downstream creation");
    unsetenv("PHOENIX_GAME_DEVICE");
    VkInstance ordinary{}; require(make(ordinary)==VK_SUCCESS,"create ordinary instance"); count=2;
    require(vkEnumeratePhysicalDevices(ordinary,&count,output)==VK_SUCCESS && count==2,"no selection preserves behavior");
    for (auto instance : {game,runtime,otherThread,missing,ordinary}) vkDestroyInstance(instance,nullptr);
    require(instances.empty(),"instance state retired");
}
