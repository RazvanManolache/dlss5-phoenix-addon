// Native model schedule through CUDA driver/runtime APIs on Linux and Windows.
#pragma once
#include "../src/native/model/bundle_loader.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "half_rgba.h"
#include "native_cuda_pixels.h"
#include "network_schedule.h"
#include "frame.h"
#include <chrono>
#include <map>
#include <cmath>

namespace phoenix_native_cuda {
using Json = nlohmann::json;
inline void check(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(result));
}
inline void driver(CUresult result) {
    if (result != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(result, &message);
        throw std::runtime_error(message ? message : "CUDA driver error");
    }
}
using Clock = std::chrono::steady_clock;
// An in-process caller may have a different device or a non-primary CUDA
// context current. Keep the exact driver context across every public call,
// including failed construction and resource destruction.
struct CurrentContextGuard {
    CUcontext previous{};
    CurrentContextGuard() { driver(cuInit(0)); driver(cuCtxGetCurrent(&previous)); }
    ~CurrentContextGuard() { (void)cuCtxSetCurrent(previous); }
    CurrentContextGuard(const CurrentContextGuard&) = delete;
    CurrentContextGuard& operator=(const CurrentContextGuard&) = delete;
};
inline double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
struct PrimaryContext {
    CUcontext value{};
    CUdevice device{};
    void retain(int ordinal){driver(cuDeviceGet(&device,ordinal));driver(cuDevicePrimaryCtxRetain(&value,device));driver(cuCtxSetCurrent(value));}
    void activate()const{driver(cuCtxSetCurrent(value));}
    ~PrimaryContext(){if(value)(void)cuDevicePrimaryCtxRelease(device);}
};
struct Resources {
    void* workspace_raw{};
    void* weight_raw{};
    void* workspace{};
    void* weights{};
    cudaArray_t input{}, output{}, history{}, motion{}, depth{}, mask{}, conditioning_motion{};
    cudaTextureObject_t color{}, prior{}, vectors{}, z{}, conditioning{}, conditioning_vectors{};
    void* repeat_counters{};
    void* retained_scratch{};
    cudaSurfaceObject_t result{}, input_surface{}, motion_surface{}, history_surface{};
    uint8_t* byte_buffer{};
    uint8_t* byte_staging{};
    uint32_t* invalid{};
    cudaStream_t stream{};
    cudaEvent_t start{}, end{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    std::vector<CUmodule> modules;
    Resources() = default;
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
    ~Resources() {
        if (stream)
            cudaStreamSynchronize(stream);
        if (executable)
            cudaGraphExecDestroy(executable);
        if (graph)
            cudaGraphDestroy(graph);
        for (auto module : modules)
            cuModuleUnload(module);
        if (start)
            cudaEventDestroy(start);
        if (end)
            cudaEventDestroy(end);
        if (color)
            cudaDestroyTextureObject(color);
        for(auto texture:{prior,vectors,z,conditioning,conditioning_vectors})if(texture)cudaDestroyTextureObject(texture);
        if(motion_surface)cudaDestroySurfaceObject(motion_surface);
        if(history_surface)cudaDestroySurfaceObject(history_surface);
        for(auto array:{history,motion,depth,mask,conditioning_motion})if(array)cudaFreeArray(array);
        if(repeat_counters)cudaFree(repeat_counters);
        if(retained_scratch)cudaFree(retained_scratch);
        if (result)
            cudaDestroySurfaceObject(result);
        if (input_surface)
            cudaDestroySurfaceObject(input_surface);
        if (byte_buffer)
            cudaFree(byte_buffer);
        if (byte_staging)
            cudaFreeHost(byte_staging);
        if (invalid)
            cudaFree(invalid);
        if (input)
            cudaFreeArray(input);
        if (output)
            cudaFreeArray(output);
        if (workspace_raw)
            cudaFree(workspace_raw);
        if (weight_raw)
            cudaFree(weight_raw);
        if (stream)
            cudaStreamDestroy(stream);
    }
    void allocate(void*& raw, void*& aligned, size_t bytes) {
        if (!bytes || bytes > SIZE_MAX - 511)
            throw std::runtime_error("Native arena extent");
        check(cudaMalloc(&raw, bytes + 511));
        aligned =
            reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(raw) + 511) & ~uintptr_t(511));
        // Initialization must precede uploads on this nonblocking stream.
        // A default-stream memset can race a later weight upload.
        check(cudaMemsetAsync(aligned, 0, bytes, stream));
    }
};
class Executor {
    int selected_ = -1;
    // Keep a context lease through all resource destruction. Selecting an
    // ordinal alone can leave another library's same-device context current.
    PrimaryContext context_;
    Resources resources_;
    struct Step {
        CUfunction function{};
        std::array<uint32_t, 3> grid, block;
        std::vector<uint8_t> bytes;
    };
    std::vector<Step> steps_;
    std::vector<Step> original_steps_;
    std::map<std::string,CUfunction> functions_;
    Json frame_plan_;
    phoenix_model::NativeLayout layout_;
    size_t counter_bytes_{};
    std::array<unsigned,8> repeats_{1,1,1,1,1,1,1,1};
    CUgraphNode stem_node_{},head_node_{};
    bool history_valid_=false;
    uint64_t sampling_=0;
    std::vector<float> sanitized_motion_;
    std::vector<float> dilated_motion_;
    std::vector<float4> rgba_;
    std::vector<__half> raw_;
    std::vector<float> rgb_;
    bool failed_ = false;

  public:
    const unsigned width, height;
    Json status;
    Executor(const Json& config, unsigned frame_width, unsigned frame_height, Executor* shared=nullptr)
        : width(frame_width), height(frame_height) {
        if (!width || !height || width > 8192 || height > 8192 ||
            uint64_t(width) * height > 16777216)
            throw std::runtime_error("Native CUDA frame extent exceeds budget");
        if (config.at("backend") != "native-cuda")
            throw std::runtime_error("Expected native-cuda backend");
        const std::string requested = config.at("device");
        if (requested.size() != 37 || requested.rfind("cuda:", 0) != 0)
            throw std::runtime_error("Native CUDA requires cuda:<32 hexadecimal UUID digits>");
        const auto validated = config.value("validated_devices", std::vector<std::string>{});
        if (!config.value("allow_unvalidated", false) &&
            std::find(validated.begin(), validated.end(), requested) == validated.end())
            throw std::runtime_error("Native CUDA device requires explicit validation opt-in");
        // Network controls belong to individual frame requests. Keep the
        // immutable session/device/model configuration explicit.
        const std::vector<std::string> allowed{"backend",           "device",
                                               "runtime_root",      "native_bundle",
                                               "allow_unvalidated", "validated_devices",
                                               "weight_source",     "local_tone",
                                               "local_structure",   "skin_structure",
                                               "auto_mask", "execution", "runtime_sha256", "files", "broker_protocol", "cpu_threads", "diagnostics", "d3d12_interop"};
        for (const auto& item : config.items())
            if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
                throw std::runtime_error("Unsupported native CUDA configuration: " + item.key());
        int count = 0;
        check(cudaGetDeviceCount(&count));
        for (int index = 0; index < count; ++index) {
            cudaDeviceProp properties{};
            check(cudaGetDeviceProperties(&properties, index));
            std::string uuid = "cuda:";
            for (unsigned char byte : properties.uuid.bytes) {
                char hex[3];
                std::snprintf(hex, sizeof(hex), "%02x", byte);
                uuid += hex;
            }
            if (uuid == requested) {
                if (selected_ >= 0)
                    throw std::runtime_error("Ambiguous CUDA UUID");
                selected_ = index;
            }
        }
        if (selected_ < 0)
            throw std::runtime_error("Requested CUDA UUID missing");
        context_.retain(selected_);
        check(cudaSetDevice(selected_));
        cudaDeviceProp gpu{};
        check(cudaGetDeviceProperties(&gpu, selected_));
        if (gpu.major != 12 || gpu.minor != 0)
            throw std::runtime_error("Native CUDA preview currently requires SM120");
        auto& s = resources_;
        check(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        auto root = phoenix_fs::from_utf8(config.at("runtime_root").get<std::string>());
        auto folder = phoenix_fs::from_utf8(
            config.value("native_bundle", phoenix_model::source_path_string(root / "native-cuda")));
        if (!root.is_absolute() || !folder.is_absolute())
            throw std::runtime_error("Native runtime/bundle paths must be absolute");
        phoenix_bundle::Bundle bundle(
            folder, phoenix_model::mapped_file_digest,
            phoenix_model::WeightSource::config(config.at("weight_source")), false, "auto", 120,
            "cuda-texture-objects/1");
        if (bundle.manifest.at("architectures") != Json::array({"sm_120"}))
            throw std::runtime_error("Native CUDA bundle architecture mismatch");
        auto frame = bundle.select_frame(width, height, [](const Json&) {});
        frame_plan_=frame;
        layout_=phoenix_model::native_layout(frame.value("model_architecture",std::string("phoenix-network37/1")));
        auto packed = shared?std::vector<uint8_t>{}:bundle.load_weights();
        const size_t weight_bytes=shared?shared->status.at("weight_bytes").get<size_t>():packed.size();
        const size_t workspace_bytes = frame.at("arenas").at("workspace").at("bytes");
        if (workspace_bytes > uint64_t(8) * 1024 * 1024 * 1024 ||
            packed.size() > uint64_t(2) * 1024 * 1024 * 1024)
            throw std::runtime_error("Native CUDA allocation budget exceeded");
        s.allocate(s.workspace_raw, s.workspace, workspace_bytes);
        if(shared)s.weights=shared->resources_.weights;
        else {s.allocate(s.weight_raw, s.weights, packed.size());
            check(cudaMemcpyAsync(s.weights, packed.data(), packed.size(), cudaMemcpyHostToDevice,s.stream));}
        const auto input_desc = cudaCreateChannelDesc<float4>();
        check(cudaMallocArray(&s.input, &input_desc, width, height, cudaArraySurfaceLoadStore));
        const cudaChannelFormatDesc output_desc{16, 16, 16, 16, cudaChannelFormatKindFloat};
        check(cudaMallocArray(&s.output, &output_desc, width, height, cudaArraySurfaceLoadStore));
        cudaResourceDesc resource{};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = s.input;
        cudaTextureDesc texture{};
        texture.addressMode[0] = texture.addressMode[1] = cudaAddressModeClamp;
        texture.filterMode = cudaFilterModePoint;
        texture.readMode = cudaReadModeElementType;
        texture.normalizedCoords = 1;
        check(cudaCreateTextureObject(&s.color, &resource, &texture, nullptr));
        check(cudaCreateSurfaceObject(&s.input_surface, &resource));
        resource.res.array.array = s.output;
        check(cudaCreateSurfaceObject(&s.result, &resource));
        auto texture_array=[&](cudaArray_t& array,cudaTextureObject_t& handle,cudaChannelFormatDesc desc,bool linear){
            check(cudaMallocArray(&array,&desc,width,height,cudaArraySurfaceLoadStore));
            cudaResourceDesc view{};view.resType=cudaResourceTypeArray;view.res.array.array=array;
            auto sampler=texture;sampler.filterMode=linear?cudaFilterModeLinear:cudaFilterModePoint;
            check(cudaCreateTextureObject(&handle,&view,&sampler,nullptr));
        };
        texture_array(s.history,s.prior,output_desc,true);
        {cudaResourceDesc view{};view.resType=cudaResourceTypeArray;view.res.array.array=s.history;
            check(cudaCreateSurfaceObject(&s.history_surface,&view));}
        texture_array(s.motion,s.vectors,cudaCreateChannelDesc<float2>(),false);
        {cudaResourceDesc view{};view.resType=cudaResourceTypeArray;view.res.array.array=s.motion;
            check(cudaCreateSurfaceObject(&s.motion_surface,&view));}
        texture_array(s.depth,s.z,cudaCreateChannelDesc<float>(),false);
        texture_array(s.mask,s.conditioning,cudaCreateChannelDesc<float4>(),false);
        texture_array(s.conditioning_motion,s.conditioning_vectors,cudaCreateChannelDesc<float2>(),false);
        std::map<std::string, CUfunction> functions=shared?shared->functions_:std::map<std::string,CUfunction>{};
        if(!shared)for (const auto& item : bundle.kernels.items()) {
            const auto& kernel = item.value();
            CUmodule module{};
            driver(cuModuleLoad(&module, kernel.at("binary").get<std::string>().c_str()));
            s.modules.push_back(module);
            std::string entry = kernel.at("entry");
            if (kernel.contains("large_input_entry") &&
                uint64_t(width) * height >= kernel.at("large_input_min_pixels").get<uint64_t>())
                entry = kernel.at("large_input_entry");
            driver(cuModuleGetFunction(&functions[item.key()], module, entry.c_str()));
        }
        functions_=functions;
        const std::array<phoenix_launch_plan::Arena, 2> arenas{
            {{reinterpret_cast<uint64_t>(s.workspace), workspace_bytes},
             {reinterpret_cast<uint64_t>(s.weights), weight_bytes}}};
        for (size_t index = 1; index + 1 < frame.at("launches").size(); ++index) {
            const auto& source = frame.at("launches")[index];
            auto plan = phoenix_bundle::launch(source, [&](const Json& binding) -> uint64_t {
                const std::string role = binding.at("role");
                if (role == "color")
                    return s.color;
                if (role == "history_previous" || role == "motion")
                    return 0;
                if (role == "neural_output_surface")
                    return s.result;
                throw std::runtime_error("Unsupported native CUDA descriptor: " + role);
            });
            steps_.push_back({functions.at(source.at("name").get<std::string>()), plan.grid,
                              plan.block, phoenix_launch_plan::materialize(plan, arenas)});
        }
        phoenix_addon::Controls controls;
        controls.reset = 1;
        controls.inverted = 0;
        controls.tone = config.value("local_tone", 1.f);
        controls.structure = config.value("local_structure", 1.f);
        controls.skin = config.value("skin_structure", -1.f);
        controls.auto_mask = config.value("auto_mask", true);
        const auto resolved =
            phoenix_addon::resolve_controls(controls, width, height, false, false, 0);
        phoenix_addon::patch_stem(steps_.front().bytes.data(), steps_.front().bytes.size(),
                                  resolved);
        phoenix_addon::patch_head(steps_.back().bytes.data(), steps_.back().bytes.size(), resolved);
        auto clear = phoenix_bundle::hex(frame.at("launches")[0].at("template_hex"));
        uint32_t words{};
        std::memcpy(&words, clear.data() + 8, sizeof(words));
        if (!words || size_t(words) * 4 > workspace_bytes)
            throw std::runtime_error("Native counter extent");
        counter_bytes_=size_t(words)*4;
        original_steps_=steps_;
        check(cudaEventCreate(&s.start));
        check(cudaEventCreate(&s.end));
        check(cudaStreamSynchronize(s.stream));
        bool capturing = false;
        try {
            check(cudaStreamBeginCapture(s.stream, cudaStreamCaptureModeThreadLocal));
            capturing = true;
            check(cudaEventRecordWithFlags(s.start,s.stream,cudaEventRecordExternal));
            check(cudaMemsetAsync(s.workspace, 0xff, size_t(words) * 4, s.stream));
            for (auto& step : steps_) {
                size_t bytes = step.bytes.size();
                void* extra[]{CU_LAUNCH_PARAM_BUFFER_POINTER, step.bytes.data(),
                              CU_LAUNCH_PARAM_BUFFER_SIZE, &bytes, CU_LAUNCH_PARAM_END};
                driver(cuLaunchKernel(step.function, step.grid[0], step.grid[1], step.grid[2],
                                      step.block[0], step.block[1], step.block[2], 0,
                                      reinterpret_cast<CUstream>(s.stream), nullptr, extra));
            }
            check(cudaEventRecordWithFlags(s.end,s.stream,cudaEventRecordExternal));
            auto result = cudaStreamEndCapture(s.stream, &s.graph);
            capturing = false;
            check(result);
            check(cudaGraphInstantiate(&s.executable, s.graph, nullptr, nullptr, 0));
            find_boundary_nodes();
        } catch (...) {
            if (capturing) {
                cudaGraph_t abandoned{};
                cudaStreamEndCapture(s.stream, &abandoned);
                if (abandoned)
                    cudaGraphDestroy(abandoned);
            }
            throw;
        }
        status = {{"backend", "native-cuda"},
                  {"state", "ready"},
                  {"device", requested},
                  {"context_policy", "owned-primary-preserve-caller"},
                  {"gpu", gpu.name},
                  {"model", bundle.weight_source.model_id},
                  {"model_resource_sha256", bundle.model_resource.sha256},
                  {"boundary_abi", "cuda-texture-objects/1"},
                  {"frame_contract", "temporal-rgb/1"},
                  {"width", width},
                  {"height", height},
                  {"neural_launches", steps_.size()},
                  {"retained_residual_copies", 0},
                  {"workspace_bytes", workspace_bytes},
                  {"weight_bytes", weight_bytes},
                  {"frames", 0},
                  {"timing_scope", "GPU counter clear and full native stem/model/head; host "
                                   "transfers reported separately"}};
    }
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    ~Executor() {
        if (context_.value)
            (void)cuCtxSetCurrent(context_.value);
    }
    void find_boundary_nodes(){
        auto& s=resources_;size_t count=0;driver(cuGraphGetNodes(reinterpret_cast<CUgraph>(s.graph),nullptr,&count));
        std::vector<CUgraphNode> nodes(count);driver(cuGraphGetNodes(reinterpret_cast<CUgraph>(s.graph),nodes.data(),&count));
        stem_node_=head_node_=nullptr;
        for(auto node:nodes){CUgraphNodeType type;driver(cuGraphNodeGetType(node,&type));if(type!=CU_GRAPH_NODE_TYPE_KERNEL)continue;
            CUDA_KERNEL_NODE_PARAMS args{};driver(cuGraphKernelNodeGetParams(node,&args));
            if(args.func==steps_.front().function)stem_node_=node;
            if(args.func==steps_.back().function)head_node_=node;
        }
        if(!stem_node_||!head_node_)throw std::runtime_error("Native CUDA boundary graph nodes missing");
    }
    void schedule(const std::array<unsigned,8>& counts){
        if(counts==repeats_)return;
        auto& s=resources_;check(cudaStreamSynchronize(s.stream));
        const bool original=counts==std::array<unsigned,8>{1,1,1,1,1,1,1,1};
        auto pointer=[](const Step& step,size_t offset){uint64_t value{};std::memcpy(&value,step.bytes.data()+offset,8);return value;};
        auto put=[](Step& step,size_t offset,uint64_t value){std::memcpy(step.bytes.data()+offset,&value,8);};
        std::vector<Step> next;
        if(original){
            // Returning from an adjusted workload restores the original arena
            // lifetimes and graph, just as a fresh full-network session does.
            // Residual preservation is necessary only for changed schedules.
            next=original_steps_;
        }else{
        if(!s.repeat_counters)check(cudaMalloc(&s.repeat_counters,counter_bytes_*25));
        uint32_t rows{},columns{};
        std::memcpy(&rows,original_steps_.at(layout_.repack_in()).bytes.data()+16,4);
        std::memcpy(&columns,original_steps_.at(layout_.repack_in()).bytes.data()+20,4);
        const size_t retained_bytes=((size_t(rows)*columns+31)/32*32)*1024;
        if(!s.retained_scratch)check(cudaMalloc(&s.retained_scratch,retained_bytes));
        const auto begin=layout_.transformer_begin(),end=layout_.repack_out();
        uint64_t retained=pointer(original_steps_.at(layout_.repack_in()),8),publication=0;
        // Skipping stages extends tensor liveness beyond the original arena
        // planner's intervals. Preserve the current residual before any later
        // stage reuses that address (including an aliased output repack).
        auto preserve=[&]{Step copy{};copy.bytes.resize(16);put(copy,0,retained);put(copy,8,retained_bytes);next.push_back(std::move(copy));retained=reinterpret_cast<uint64_t>(s.retained_scratch);};
        for(unsigned i=0;i<original_steps_.size();++i){
            if(i==begin){
                for(unsigned block=0;block<8;++block){
                    if(counts[block]>4)throw std::runtime_error("Native CUDA repetitions outside 0..4");
                    if(!counts[block])continue;
                    preserve();
                    std::array<Step,5> group;
                    for(unsigned stage=0;stage<5;++stage)group[stage]=original_steps_.at(begin+block*5+stage);
                    put(group[0],0,retained);put(group[1],8,retained);
                    put(group[0],48,publication?publication:reinterpret_cast<uint64_t>(s.repeat_counters)+counter_bytes_*24);
                    for(auto& step:group)next.push_back(step);
                    retained=pointer(group[4],16);publication=pointer(group[4],56);
                    for(unsigned repeat=1;repeat<counts[block];++repeat){
                        preserve();
                        const uint64_t bank=reinterpret_cast<uint64_t>(s.repeat_counters)+counter_bytes_*(block*3+repeat-1);
                        for(unsigned stage=0;stage<5;++stage){
                            auto step=group[stage];if(stage==0)put(step,0,retained);if(stage==1)put(step,8,retained);
                            const auto& source=frame_plan_.at("launches").at(begin+block*5+stage+1);
                            for(const auto& relocation:source.at("relocations")){
                                const std::string role=relocation.at("role");
                                const bool synchronization=role.starts_with("wait")||role.starts_with("publication")||role.starts_with("reduction_counter")||role=="counter"||role=="synchronization_arena";
                                if(relocation.at("relative_offset").is_null()||!synchronization||(stage==0&&role=="wait_optional"))continue;
                                const auto offset=relocation.at("relative_offset").get<uint64_t>();
                                if(relocation.at("arena")!="workspace"||offset>=counter_bytes_)throw std::runtime_error("Native repeat synchronization outside prefix");
                                put(step,relocation.at("argument_offset").get<unsigned>(),bank+offset);
                            }
                            next.push_back(std::move(step));
                        }
                        retained=pointer(group[4],16);
                    }
                }
                i=end;
            }
            auto step=original_steps_.at(i);if(i==end){preserve();put(step,0,retained);}next.push_back(std::move(step));
        }
        }
        steps_=std::move(next);
        if(s.executable){check(cudaGraphExecDestroy(s.executable));s.executable=nullptr;}
        if(s.graph){check(cudaGraphDestroy(s.graph));s.graph=nullptr;}
        check(cudaStreamBeginCapture(s.stream,cudaStreamCaptureModeThreadLocal));
        try{
            check(cudaEventRecordWithFlags(s.start,s.stream,cudaEventRecordExternal));
            check(cudaMemsetAsync(s.workspace,0xff,counter_bytes_,s.stream));
            if(!original){
                check(cudaMemsetAsync(s.repeat_counters,0xff,counter_bytes_*24,s.stream));
                check(cudaMemsetAsync(static_cast<uint8_t*>(s.repeat_counters)+counter_bytes_*24,0,counter_bytes_,s.stream));
            }
            for(auto& step:steps_){
                if(!step.function){check(cudaMemcpyAsync(s.retained_scratch,reinterpret_cast<void*>(pointer(step,0)),size_t(pointer(step,8)),cudaMemcpyDeviceToDevice,s.stream));continue;}
                size_t bytes=step.bytes.size();void* extra[]{CU_LAUNCH_PARAM_BUFFER_POINTER,step.bytes.data(),CU_LAUNCH_PARAM_BUFFER_SIZE,&bytes,CU_LAUNCH_PARAM_END};
                driver(cuLaunchKernel(step.function,step.grid[0],step.grid[1],step.grid[2],step.block[0],step.block[1],step.block[2],0,reinterpret_cast<CUstream>(s.stream),nullptr,extra));}
            check(cudaEventRecordWithFlags(s.end,s.stream,cudaEventRecordExternal));
            check(cudaStreamEndCapture(s.stream,&s.graph));check(cudaGraphInstantiate(&s.executable,s.graph,nullptr,nullptr,0));find_boundary_nodes();
        }catch(...){cudaGraph_t abandoned{};cudaStreamEndCapture(s.stream,&abandoned);if(abandoned)cudaGraphDestroy(abandoned);throw;}
        repeats_=counts;
    }
    void prepare_frame(bool reset,const float* motion,const float* depth,const float* validity,const float* mask,
                       const phoenix_cpp::Controls& tuning,bool reversed,const std::array<unsigned,8>& repeats,
                       cudaTextureObject_t gpu_motion=0,cudaSurfaceObject_t gpu_output=0,float motion_x=1,float motion_y=1){
        if(failed_)throw std::runtime_error("Native CUDA session failed; recreate it");
        context_.activate();auto& s=resources_;schedule(repeats);
        auto upload=[&](cudaArray_t target,const float* data,unsigned channels){if(data)check(cudaMemcpy2DToArrayAsync(target,0,0,data,size_t(width)*channels*4,size_t(width)*channels*4,height,cudaMemcpyHostToDevice,s.stream));};
        if(motion&&validity){sanitized_motion_.assign(motion,motion+size_t(width)*height*2);
            for(size_t p=0;p<size_t(width)*height;++p)if(validity[p]==0)sanitized_motion_[p*2]=float(width)*2;
            motion=sanitized_motion_.data();}
        const bool masked_depth=motion&&depth&&validity;
        if(masked_depth){
            // Match the shared contract: invalid center pixels never use
            // history, and invalid neighbours cannot win depth dilation.
            dilated_motion_.resize(size_t(width)*height*2);
            for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x){
                const size_t pixel=size_t(y)*width+x;size_t closest=pixel;
                if(validity[pixel])for(int dy:{-1,1})for(int dx:{-1,1}){
                    const auto nx=unsigned(std::clamp(int(x)+dx,0,int(width)-1));
                    const auto ny=unsigned(std::clamp(int(y)+dy,0,int(height)-1));const size_t candidate=size_t(ny)*width+nx;
                    if(validity[candidate]&&(reversed?depth[candidate]>depth[closest]:depth[candidate]<depth[closest]))closest=candidate;
                }
                dilated_motion_[pixel*2]=validity[pixel]?motion[closest*2]:float(width)*2;
                dilated_motion_[pixel*2+1]=validity[pixel]?motion[closest*2+1]:0.f;
            }
            upload(s.conditioning_motion,dilated_motion_.data(),2);
        }
        upload(s.motion,motion,2);upload(s.depth,depth,1);upload(s.mask,mask,4);
        phoenix_addon::Controls controls;controls.reset=reset;controls.inverted=reversed;
        controls.tone=tuning.tone;controls.structure=tuning.structure;controls.skin=tuning.skin;controls.auto_mask=tuning.auto_mask;
        controls.motion_x=motion_x;controls.motion_y=motion_y;
        const auto resolved=phoenix_addon::resolve_controls(controls,width,height,motion!=nullptr||gpu_motion!=0,history_valid_,reset?0:sampling_);
        auto stem=original_steps_.front().bytes,head=original_steps_.back().bytes;
        if(resolved.recurrent){const auto vectors=gpu_motion?gpu_motion:s.vectors;
            phoenix_addon::put(stem.data(),8,uint64_t(s.prior));phoenix_addon::put(stem.data(),16,uint64_t(masked_depth?s.conditioning_vectors:vectors));
            phoenix_addon::put(head.data(),88,uint64_t(s.prior));phoenix_addon::put(head.data(),96,uint64_t(vectors));}
        if(gpu_output)phoenix_addon::put(head.data(),16,uint64_t(gpu_output));
        auto patched=resolved;patched.control_mask_handle=mask?s.conditioning:0;
        phoenix_addon::patch_stem(stem.data(),stem.size(),patched);phoenix_addon::patch_head(head.data(),head.size(),patched);
        if(depth&&!masked_depth){phoenix_addon::put(stem.data(),24,uint64_t(s.z));phoenix_addon::put(stem.data(),88,resolved.color);}
        auto update=[&](CUgraphNode node,Step& step,std::vector<uint8_t>& bytes){
            if(step.bytes==bytes)return;
            step.bytes=std::move(bytes);size_t size=step.bytes.size();
            void* extra[]{CU_LAUNCH_PARAM_BUFFER_POINTER,step.bytes.data(),CU_LAUNCH_PARAM_BUFFER_SIZE,&size,CU_LAUNCH_PARAM_END};
            CUDA_KERNEL_NODE_PARAMS args{};args.func=step.function;args.gridDimX=step.grid[0];args.gridDimY=step.grid[1];args.gridDimZ=step.grid[2];
            args.blockDimX=step.block[0];args.blockDimY=step.block[1];args.blockDimZ=step.block[2];args.extra=extra;
            driver(cuGraphExecKernelNodeSetParams(reinterpret_cast<CUgraphExec>(s.executable),node,&args));};
        update(stem_node_,steps_.front(),stem);update(head_node_,steps_.back(),head);
        status["history_used"]=resolved.recurrent;status["sampling_index"]=reset?0:sampling_;
        status["neural_launches"]=std::count_if(steps_.begin(),steps_.end(),[](const Step& step){return step.function!=nullptr;});
        status["retained_residual_copies"]=std::count_if(steps_.begin(),steps_.end(),[](const Step& step){return step.function==nullptr;});
    }
    int device_ordinal()const{return selected_;}
    cudaStream_t stream()const{return resources_.stream;}
    void activate(){if(failed_)throw std::runtime_error("Native CUDA session failed; recreate it");context_.activate();}
    void poison(){failed_=true;status["state"]="failed";}
    void mark_committed(){history_valid_=true;sampling_=status.value("sampling_index",uint64_t(0))+1;}
    void commit(cudaArray_t source=nullptr){
        context_.activate();auto& s=resources_;
        CUDA_MEMCPY2D copy{};copy.srcMemoryType=CU_MEMORYTYPE_ARRAY;copy.srcArray=reinterpret_cast<CUarray>(source?source:s.output);
        copy.dstMemoryType=CU_MEMORYTYPE_ARRAY;copy.dstArray=reinterpret_cast<CUarray>(s.history);
        copy.WidthInBytes=size_t(width)*8;copy.Height=height;
        driver(cuMemcpy2DAsync(&copy,reinterpret_cast<CUstream>(s.stream)));
        // The public host API completes its history transaction before return.
        // Device-to-device synchronous-named copies need not block the host.
        check(cudaStreamSynchronize(s.stream));
        mark_committed();
    }
    void enqueue_external_finish(cudaTextureObject_t original,cudaSurfaceObject_t alpha,cudaSurfaceObject_t output,float intensity,unsigned style,float tone){
        auto& s=resources_;
        check(finish_external(original,alpha,output,s.history_surface,s.invalid,width,height,intensity,style,tone,s.stream));
    }
    cudaTextureObject_t device_motion()const{return resources_.vectors;}
    template<class Finish>
    void run_external(cudaTextureObject_t input,cudaTextureObject_t motion,cudaSurfaceObject_t output,float motion_x,float motion_y,Finish&& finish){
        if(failed_)throw std::runtime_error("Native CUDA session failed; recreate it");
        context_.activate();auto& s=resources_;
        if(!s.invalid)check(cudaMalloc(&s.invalid,sizeof(uint32_t)));
        auto invalid=[&]{uint32_t flag{};check(cudaMemcpyAsync(&flag,s.invalid,sizeof(flag),cudaMemcpyDeviceToHost,s.stream));check(cudaStreamSynchronize(s.stream));return flag;};
        auto started=Clock::now();check(cudaMemsetAsync(s.invalid,0,sizeof(uint32_t),s.stream));
        check(unpack_external(input,motion,s.input_surface,s.motion_surface,s.invalid,width,height,motion_x,motion_y,s.stream));
        // Unpacking sanitizes rejected inputs before the graph consumes them.
        // Neither history nor caller output is published until both checks pass,
        // so input and output validation can share one completion/readback.
        const double prepare_ms=milliseconds(started);started=Clock::now();
        uint32_t flags{};
        try{
            check(cudaGraphLaunch(s.executable,s.stream));
            check(validate_external(output,s.invalid,width,height,s.stream));
            finish();
            flags=invalid();
            if(!(flags&1u)&&(flags&2u))throw std::runtime_error("Nonfinite native GPU output");
            float gpu_ms{};check(cudaEventElapsedTime(&gpu_ms,s.start,s.end));
            status.update({{"state","completed"},{"operator_ms",gpu_ms},{"prepare_ms",prepare_ms},
                {"execute_with_transfers_ms",milliseconds(started)},{"resolve_ms",0.},{"timing_intervals",1},
                {"neural_host_transfer_bytes",0},{"pixel_transport","d3d12-cuda-shared-textures"}});
        }catch(...){failed_=true;status["state"]="failed";throw;}
        if(flags&1u)throw std::invalid_argument("Nonfinite D3D12 color or motion");
    }
    const uint8_t* run_rgba8(const uint8_t* bytes, size_t stride, bool bgra) {
        if (failed_)
            throw std::runtime_error("Native CUDA session failed; recreate it");
        context_.activate();
        auto& s = resources_;
        try {
            const size_t count = size_t(width) * height * 4;
            if (!s.byte_buffer) {
                check(cudaMalloc(&s.byte_buffer, count));
                if(!s.invalid)check(cudaMalloc(&s.invalid, sizeof(uint32_t)));
                check(cudaHostAlloc(&s.byte_staging, count, cudaHostAllocDefault));
            }
            auto started = Clock::now();
            for (unsigned y = 0; y < height; ++y)
                std::memcpy(s.byte_staging + size_t(y) * width * 4, bytes + size_t(y) * stride,
                            size_t(width) * 4);
            const double prepare_ms = milliseconds(started);
            started = Clock::now();
            check(cudaMemcpyAsync(s.byte_buffer, s.byte_staging, count, cudaMemcpyHostToDevice,
                                  s.stream));
            check(unpack_rgba8(s.byte_buffer, s.input_surface, width, height, bgra, s.stream));
            check(cudaGraphLaunch(s.executable, s.stream));
            check(cudaMemsetAsync(s.invalid, 0, sizeof(uint32_t), s.stream));
            check(pack_rgba8(s.result, s.byte_buffer, s.invalid, width, height, bgra, s.stream));
            check(cudaMemcpyAsync(s.byte_staging, s.byte_buffer, count, cudaMemcpyDeviceToHost,
                                  s.stream));
            uint32_t invalid{};
            check(cudaMemcpyAsync(&invalid, s.invalid, sizeof(invalid), cudaMemcpyDeviceToHost,
                                  s.stream));
            check(cudaStreamSynchronize(s.stream));
            if (invalid)
                throw std::runtime_error("Nonfinite native output");
            const double transfers_ms = milliseconds(started);
            float gpu_ms{};
            check(cudaEventElapsedTime(&gpu_ms, s.start, s.end));
            status.update({{"state", "completed"},
                           {"operator_ms", gpu_ms},
                           {"prepare_ms", prepare_ms},
                           {"execute_with_transfers_ms", transfers_ms},
                           {"resolve_ms", 0.0},
                           {"frame_api", "bounded-rgba8/1"},
                           {"pixel_transport", "bounded-rgba8"},
                           {"neural_host_transfer_bytes", uint64_t(width)*height*8},
                           {"frames", status.at("frames").get<uint64_t>() + 1}});
            return s.byte_staging;
        } catch (...) {
            failed_ = true;
            status["state"] = "failed";
            throw;
        }
    }
    const std::vector<float>& run(const float* color) {
        if (failed_)
            throw std::runtime_error("Native CUDA session failed; recreate it");
        context_.activate();
        // Packed presentation never needs these host float planes. Allocate
        // them only when an explicit bounded float V2 caller uses that route.
        if (rgba_.empty()) {
            rgba_.resize(size_t(width) * height);
            raw_.resize(size_t(width) * height * 4);
            rgb_.resize(size_t(width) * height * 3);
        }
        auto started = Clock::now();
        for (size_t pixel = 0; pixel < rgba_.size(); ++pixel) {
            const float r = color[pixel * 3], g = color[pixel * 3 + 1], b = color[pixel * 3 + 2];
            if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b))
                throw std::runtime_error("Native color must be finite");
            rgba_[pixel] = make_float4(r, g, b, 1.f);
        }
        const double prepare_ms = milliseconds(started);
        auto& s = resources_;
        try {
            started = Clock::now();
            check(cudaMemcpy2DToArrayAsync(s.input, 0, 0, rgba_.data(), width * 16, width * 16,
                                           height, cudaMemcpyHostToDevice, s.stream));
            check(cudaGraphLaunch(s.executable, s.stream));
            check(cudaMemcpy2DFromArrayAsync(raw_.data(), width * 8, s.output, 0, 0, width * 8,
                                             height, cudaMemcpyDeviceToHost, s.stream));
            check(cudaStreamSynchronize(s.stream));
            const double transfers_ms = milliseconds(started);
            float gpu_ms{};
            check(cudaEventElapsedTime(&gpu_ms, s.start, s.end));
            started = Clock::now();
            static_assert(sizeof(__half) == sizeof(uint16_t));
            half_rgba_to_rgb(reinterpret_cast<const uint16_t*>(raw_.data()), rgb_.data(),
                             rgba_.size());
            for (const float value : rgb_)
                if (!std::isfinite(value))
                    throw std::runtime_error("Nonfinite native output");
            status.update({{"state", "completed"},
                           {"operator_ms", gpu_ms},
                           {"prepare_ms", prepare_ms},
                           {"execute_with_transfers_ms", transfers_ms},
                           {"resolve_ms", milliseconds(started)},
                           {"timing_intervals", 1},
                           {"frames", status.at("frames").get<uint64_t>() + 1}});
            return rgb_;
        } catch (...) {
            failed_ = true;
            status["state"] = "failed";
            throw;
        }
    }
};
} // namespace phoenix_native_cuda
