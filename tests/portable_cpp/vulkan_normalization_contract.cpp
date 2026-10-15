// FP32 Q/K squares expose mobile FMA rounding that half-operand GEMM tests miss.
#include "../../native_runtime/backend.h"
#include "../../native_runtime/backend_options.h"
#include "../../native_runtime/host_math.h"
#include "../../native_runtime/platform.h"
#include <iostream>
#include <cstring>
using namespace phoenix_cpp;
int main(int argc,char** argv) {
    try {
        demand(argc==3,"Usage: vulkan_normalization_contract RUNTIME DEVICE");
        auto root=std::filesystem::absolute(argv[1]);
        auto module=open_library(root/library_name("phoenix_vulkan_cpp"));
        demand(module,"Vulkan backend unavailable");
        auto create=reinterpret_cast<Backend*(*)(const char*,const char*,uint32_t,char*,size_t)>(
            library_symbol(module,"phoenix_backend_create_utf8"));
        auto destroy=reinterpret_cast<DestroyBackend>(library_symbol(module,"phoenix_backend_destroy"));
        demand(create&&destroy,"Configured backend factory unavailable");
        auto folder=(root/"vulkan").string();char error[2048]{};
        std::unique_ptr<Backend,DestroyBackend> backend(create(argv[2],folder.c_str(),
            VulkanWorkgroupCompatibility|VulkanTiledFallback,error,sizeof(error)),destroy);
        demand(bool(backend),error);
        uint32_t random=1039;size_t checked=0;
        for(unsigned rows:{1u,17u,129u,8193u})for(unsigned heads:{1u,16u})for(bool global:{false,true}) {
            std::vector<float> input(size_t(rows)*32),gains(heads);
            for(size_t i=0;i<input.size();++i){random^=random<<13;random^=random>>17;random^=random<<5;
                input[i]=(i/32)%17==0?0.f:HostMath::value((random&0x807fffffu)|0x3f000000u);}
            for(unsigned h=0;h<heads;++h)gains[h]=.713411f+h*.014131f;
            auto a=backend->allocate(input.size()*4),scale=backend->allocate(gains.size()*4),out=backend->allocate(input.size()*2);
            backend->upload(a,input.data(),input.size()*4);backend->upload(scale,gains.data(),gains.size()*4);
            for(unsigned query:{0u,1u}) {
                phx::Operation op{};op.family=phx::RowReduction;op.flags=phx::StoreHalf|(global?phx::GlobalAttention:0u);
                op.left_input=a.address;op.auxiliary=scale.address;op.output=out.address;
                op.row_count=rows;op.column_count=64;op.batch_count=heads;op.reduction_count=query;
                std::vector<Event> events;backend->execute({{op,rows,0,0}},events);
                std::vector<uint16_t> actual(input.size()),expected(input.size());
                backend->download(out,actual.data(),actual.size()*2);
                op.left_input=input.data();op.auxiliary=gains.data();op.output=expected.data();
                for(unsigned row=0;row<rows;++row)phx::Kernel<HostMath>{op}.norm(row);
                demand(actual==expected,"Mobile FP32 normalization differs from CPU reference");
                checked+=actual.size();
            }
            backend->release(out);backend->release(scale);backend->release(a);
        }
        std::cout<<checked<<" normalization channels exact (FP32 inputs, zero norms, Q/K, local/global, head counts and tails)\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
