// Resident mobile GEMM tuning: ordered arithmetic, warm GPU timings and output fingerprints.
#include "../../native_runtime/backend.h"
#include "../../native_runtime/backend_options.h"
#include "../../native_runtime/platform.h"
#include <algorithm>
#include <iostream>
using namespace phoenix_cpp;
int main(int argc, char** argv) {
    try {
        demand(argc == 3, "Usage: vulkan_mobile_benchmark RUNTIME DEVICE");
        auto root = std::filesystem::absolute(argv[1]);
        auto module = open_library(root / library_name("phoenix_vulkan_cpp"));
        demand(module, "Backend missing");
        auto create = reinterpret_cast<Backend* (*)(const char*, const char*, uint32_t, char*, size_t)>(
            library_symbol(module, "phoenix_backend_create_utf8"));
        auto destroy = reinterpret_cast<DestroyBackend>(library_symbol(module, "phoenix_backend_destroy"));
        demand(create && destroy, "Backend ABI missing");
        char error[2048]{};
        const auto shaders = (root / "vulkan").string();
        std::unique_ptr<Backend, DestroyBackend> backend(
            create(argv[2], shaders.c_str(), VulkanWorkgroupCompatibility | VulkanTiledFallback |
                   VulkanCoalescedFallback | DispatchProfiling | OrderedAttention, error, sizeof error), destroy);
        demand(bool(backend), error);
        struct Shape { unsigned m,n,k,batch; bool packed,transposed; };
        for (auto s : {Shape{8192,128,32,1,true,true}, Shape{8192,32,128,1,true,true},
                       Shape{2048,1024,256,1,true,true}, Shape{640,256,1024,1,true,true},
                       Shape{64,64,32,128,false,true}, Shape{64,32,64,128,false,false}}) {
            std::vector<uint16_t> a(size_t(s.m)*s.k*s.batch), b(size_t(s.n)*s.k*s.batch);
            std::vector<uint8_t> weights(size_t(s.n)*s.k*s.batch);
            std::vector<float> result(size_t(s.m)*s.n*s.batch);
            for (size_t i=0;i<a.size();++i) a[i]=uint16_t(0x3000u+((i*37u)%0x1000u));
            for (size_t i=0;i<b.size();++i) { b[i]=uint16_t(0x2800u+((i*53u)%0x1800u)); weights[i]=uint8_t((24+i*17%65)|((i&1)?128:0)); }
            auto left=backend->allocate(a.size()*2), right=backend->allocate(s.packed?weights.size():b.size()*2), out=backend->allocate(result.size()*4);
            backend->upload(left,a.data(),a.size()*2);
            backend->upload(right,s.packed?static_cast<void*>(weights.data()):static_cast<void*>(b.data()),right.bytes);
            phx::Operation op{};
            op.family=phx::MatrixMultiply; op.flags=(s.transposed?1u:0u)|(s.packed?phx::RightInputFp8:0u);
            op.left_input=left.address;op.right_input=right.address;op.output=out.address;
            op.row_count=s.m;op.column_count=s.n;op.reduction_count=s.k;op.batch_count=s.batch;
            op.left_row_stride=s.k;op.right_row_stride=s.transposed?s.k:s.n;op.output_row_stride=s.n;
            op.left_batch_stride=s.m*s.k;op.right_batch_stride=s.n*s.k;op.output_batch_stride=s.m*s.n;
            std::vector<double> samples;
            for (unsigned i=0;i<5;++i) {
                std::vector<Event> events;backend->execute({{op,result.size(),0,0}},events);backend->wait();
                double ms=0;for(auto e:events)ms+=e.milliseconds;
                if(i>=2)samples.push_back(ms);
            }
            backend->download(out,result.data(),out.bytes);
            uint64_t hash=14695981039346656037ull;
            for(size_t i=0;i<result.size()*4;++i)hash=(hash^reinterpret_cast<const uint8_t*>(result.data())[i])*1099511628211ull;
            std::sort(samples.begin(),samples.end());
            std::cout<<"{\"m\":"<<s.m<<",\"n\":"<<s.n<<",\"k\":"<<s.k<<",\"batch\":"<<s.batch
                <<",\"packed\":"<<(s.packed?"true":"false")<<",\"gpu_ms\":"<<samples[1]<<",\"fingerprint\":\""<<std::hex<<hash<<std::dec<<"\"}"<<std::endl;
            backend->release(out);backend->release(right);backend->release(left);
        }
        backend.reset();close_library(module);return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
