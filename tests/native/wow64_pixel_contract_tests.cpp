#include "../../src/integration/compatibility/wow64/model_contract.h"
#include <iostream>
#include <stdexcept>
int main()try {
    using namespace phoenix_wow64;unsigned checks=0;
    auto check=[&](bool okay){++checks;if(!okay)throw std::runtime_error("Pixel transport contract failed");};
    check(!declared_rgba8_gpu_transport(0,1,sizeof(ModelChannel)));
    check(!declared_rgba8_gpu_transport(1,0,sizeof(ModelChannel)));
    check(!declared_rgba8_gpu_transport(2,1,sizeof(ModelChannel)));
    check(!declared_rgba8_gpu_transport(1,1,model_channel_legacy_bytes));
    check(declared_rgba8_gpu_transport(1,1,sizeof(ModelChannel)));
    ModelChannel p{};p.bytes=sizeof(p);p.base.width=17;p.base.height=3;p.base.color=0x10000;p.base.output=0x20000;
    p.base.plane_bytes=17*3*8;check(valid_model_frame_buffers(p));
    p.pixel_transport=1;check(!valid_model_frame_buffers(p));
    p.base.plane_bytes=17*3*4;check(valid_model_frame_buffers(p));
    p.backend=1;check(!valid_model_frame_buffers(p));p.backend=3;
    p.pixel_transport=2;check(!valid_model_frame_buffers(p));p.pixel_transport=1;
    p.base.output=UINT32_MAX-4;check(!valid_model_frame_buffers(p));p.base.output=p.base.color+4;check(!valid_model_frame_buffers(p));
    p.base.output=0x20000;p.base.width=8193;check(!valid_model_frame_buffers(p));p.base.width=17;
    // A legacy peer never reads the extension, even if adjacent storage is nonzero.
    p.bytes=model_channel_legacy_bytes;p.pixel_transport=99;p.base.plane_bytes=17*3*8;check(valid_model_frame_buffers(p));
    p.bytes=sizeof(p);p.pixel_transport=0;p.base.plane_bytes=17*3*8;
    check(valid_model_bootstrap_buffers(p));
    p.pixel_transport=1;p.base.color=p.base.output=0;p.base.plane_bytes=17*3*4;
    check(valid_model_bootstrap_buffers(p));check(!valid_model_frame_buffers(p));
    p.base.output=0x20000;check(!valid_model_bootstrap_buffers(p));p.base.output=0;
    p.backend=1;check(!valid_model_bootstrap_buffers(p));p.backend=3;
    p.base.plane_bytes=17*3*8;check(!valid_model_bootstrap_buffers(p));
    p.bytes=1;check(!valid_model_frame_buffers(p));
    std::cout<<"{\"passed\":true,\"cpu_only\":true,\"checks\":"<<checks<<"}\n";
}catch(const std::exception& e){std::cerr<<e.what();return 1;}
