// Layout/tail/guard checks independent of approximate-image quality tests.
#include "../../native_runtime/backend.h"
#include "../../native_runtime/backend_options.h"
#include "../../native_runtime/platform.h"
#include "../../native_runtime/host_math.h"
#include <iostream>
#include <cmath>
using namespace phoenix_cpp;
int main(int argc,char** argv){try{
    demand(argc==3,"Usage: vulkan_fast_contract RUNTIME_ROOT DEVICE");auto root=std::filesystem::absolute(argv[1]);
    auto library=open_library(root/library_name("phoenix_vulkan_cpp"));demand(library,"Backend unavailable");
    auto create=reinterpret_cast<Backend*(*)(const char*,const char*,uint32_t,char*,size_t)>(library_symbol(library,"phoenix_backend_create_utf8"));auto destroy=reinterpret_cast<DestroyBackend>(library_symbol(library,"phoenix_backend_destroy"));demand(create&&destroy,"Factory missing");
    char error[2048]{};auto shaders=(root/"vulkan").string();
    std::unique_ptr<Backend,DestroyBackend> gpu(create(argv[2],shaders.c_str(),VulkanFastMath|VulkanTiledFallback|VulkanCoalescedFallback,error,sizeof error),destroy);demand(bool(gpu),error);
    struct Shape{unsigned m,n,k,batch;bool transposed,packed;};unsigned cases=0;phx::Kernel<HostMath> math{};
    for(auto s:{Shape{1,1,3,1,true,true},Shape{35,37,33,2,true,true},Shape{17,9,65,3,true,false},Shape{31,5,16,2,false,false},Shape{65,33,255,1,true,true}}){
        unsigned lda=s.k+3,ldb=(s.transposed?s.k:s.n)+5,ldc=s.n+7;
        unsigned sa=s.m*lda+8,sb=(s.transposed?s.n:s.k)*ldb+8,sc=s.m*ldc+8;
        std::vector<uint16_t> a(size_t(sa)*s.batch),bh(size_t(sb)*s.batch);std::vector<uint8_t> b8(bh.size());std::vector<float> bv(bh.size());
        const uint8_t codes[]={0,0x18,0x98,0x20,0xa0};const float values[]={0,.0625f,-.0625f,.125f,-.125f};
        for(size_t i=0;i<a.size();++i)a[i]=uint16_t(math.half_bits(float(int(i*5%7)-3)/16));
        for(size_t i=0;i<bh.size();++i){bv[i]=values[i%5];bh[i]=uint16_t(math.half_bits(bv[i]));b8[i]=codes[i%5];}
        std::vector<float> output(size_t(sc)*s.batch,-1024),expected=output;
        auto aa=gpu->allocate(a.size()*2),bb=gpu->allocate(s.packed?b8.size():bh.size()*2),cc=gpu->allocate(output.size()*4);
        gpu->upload(aa,a.data(),aa.bytes);gpu->upload(bb,s.packed?static_cast<void*>(b8.data()):static_cast<void*>(bh.data()),bb.bytes);gpu->upload(cc,output.data(),cc.bytes);
        phx::Operation op{};op.family=phx::MatrixMultiply;op.flags=(s.transposed?1u:0u)|(s.packed?phx::RightInputFp8:0u);
        op.left_input=aa.address;op.right_input=bb.address;op.output=cc.address;op.row_count=s.m;op.column_count=s.n;op.reduction_count=s.k;op.batch_count=s.batch;
        op.left_row_stride=lda;op.right_row_stride=ldb;op.output_row_stride=ldc;op.left_batch_stride=sa;op.right_batch_stride=sb;op.output_batch_stride=sc;
        std::vector<Event> events;gpu->execute({{op,uint64_t(s.m)*s.n*s.batch,0,0}},events);gpu->download(cc,output.data(),cc.bytes);
        for(unsigned batch=0;batch<s.batch;++batch)for(unsigned row=0;row<s.m;++row)for(unsigned col=0;col<s.n;++col){float sum=0;for(unsigned k=0;k<s.k;++k)sum=std::fma(math.half_value(a[batch*sa+row*lda+k]),bv[batch*sb+(s.transposed?col*ldb+k:k*ldb+col)],sum);expected[batch*sc+row*ldc+col]=sum;}
        // These binary-grid fixtures are exactly representable in half even
        // with partial sums; any discrepancy therefore indicates a layout bug.
        for(size_t i=0;i<output.size();++i)demand(std::isfinite(output[i])&&output[i]==expected[i],"Fast GEMM tail/stride/guard mismatch");
        gpu->release(cc);gpu->release(bb);gpu->release(aa);++cases;
    }
    gpu.reset();close_library(library);std::cout<<"PASS "<<cases<<" odd/tail/transposed/strided-batch matrix cases, including padding guards.\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
