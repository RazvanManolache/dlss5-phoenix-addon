#define main phoenix_existing_probe_main
#include "four_model_probe.cpp"
#undef main
#include "../../native_runtime/portable_d3d12_pixels.h"
#include "../../native_runtime/portable_pixels.h"

int main(){try{
    constexpr unsigned w=257,h=257;const size_t count=size_t(w)*h;
    Gpu gpu(true);auto color=texture(gpu.device.Get(),w,h,16,false),motion=texture(gpu.device.Get(),w,h,16,false,2),
        output=texture(gpu.device.Get(),w,h,16,true);
    std::vector<uint16_t> rgba(count*4),mv(count*2),actual(count*4),expected;
    std::vector<float> rgb(count*3),vectors(count*2),result(count*3);
    phx::Kernel<HostMath> math{};
    for(size_t p=0;p<count;++p){
        for(unsigned c=0;c<4;++c)rgba[p*4+c]=uint16_t(p+c*171);
        mv[p*2]=uint16_t(p);mv[p*2+1]=uint16_t(p+357);
        const unsigned bits=unsigned(p)%0x7bfe;
        const float value=(math.half_value(bits)+math.half_value(bits+1))*.5f;
        result[p*3]=value;result[p*3+1]=-value;result[p*3+2]=std::nextafter(value,INFINITY);
    }
    auto upload=[&](Texture& image,const uint16_t* data,unsigned channels){
        void* mapped=nullptr;D3D12_RANGE empty{};check(image.transfer->Map(0,&empty,&mapped));
        for(unsigned y=0;y<h;++y)std::memcpy(static_cast<char*>(mapped)+image.footprint.Offset+size_t(y)*image.footprint.Footprint.RowPitch,
            data+size_t(y)*w*channels,size_t(w)*channels*2);
        image.transfer->Unmap(0,nullptr);gpu.begin();
        D3D12_TEXTURE_COPY_LOCATION from{image.transfer.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,image.footprint},
            to{image.resource.Get(),D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0};
        gpu.command->CopyTextureRegion(&to,0,0,0,&from,nullptr);
        barrier(gpu.command.Get(),image.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);gpu.submit();
    };
    upload(color,rgba.data(),4);upload(motion,mv.data(),2);
    phoenix_portable_pixels::D3D12Codec codec(gpu.device.Get(),w,h,{});
    auto equal=[](float a,float b){return std::isnan(a)?std::isnan(b):std::bit_cast<uint32_t>(a)==std::bit_cast<uint32_t>(b);};
    for(unsigned trial=0;trial<3;++trial){
        const bool has_motion=trial!=1;const float sx=trial==2?-.125f:.375f,sy=trial==2?.875f:-.625f;
        gpu.begin();codec.download(gpu.command.Get(),color.resource.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            has_motion?motion.resource.Get():nullptr,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,sx,sy);gpu.submit();
        for(size_t p=0;p<count;++p)for(unsigned c=0;c<3;++c)rgb[p*3+c]=std::clamp(math.half_value(rgba[p*4+c]),0.f,1.f);
        for(size_t i=0;i<mv.size();++i)vectors[i]=math.half_value(mv[i])*(i%2?sy:sx);
        codec.planes([&](const float* input,const float* motion){
            for(size_t i=0;i<rgb.size();++i)if(!equal(rgb[i],input[i]))throw std::runtime_error("GPU color codec differs at "+std::to_string(i)+" expected bits "+std::to_string(std::bit_cast<uint32_t>(rgb[i]))+" actual "+std::to_string(std::bit_cast<uint32_t>(input[i])));
            require(bool(motion)==has_motion,"Stale motion presence");
            if(motion)for(size_t i=0;i<vectors.size();++i)require(equal(vectors[i],motion[i]),"GPU motion codec differs");
        });
        expected=rgba;
        for(size_t p=0;p<count;++p)for(unsigned c=0;c<3;++c)expected[p*4+c]=uint16_t(math.half_bits(result[p*3+c]));
        gpu.begin();codec.upload_output(gpu.command.Get(),result.data(),output.resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS);gpu.submit();
        gpu.begin();barrier(gpu.command.Get(),output.resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION from{output.resource.Get(),D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0},
            to{output.transfer.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,output.footprint};
        gpu.command->CopyTextureRegion(&to,0,0,0,&from,nullptr);
        barrier(gpu.command.Get(),output.resource.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);gpu.submit();
        void* mapped=nullptr;D3D12_RANGE range{0,size_t(output.bytes)};check(output.transfer->Map(0,&range,&mapped));
        for(unsigned y=0;y<h;++y)std::memcpy(actual.data()+size_t(y)*w*4,static_cast<char*>(mapped)+output.footprint.Offset+size_t(y)*output.footprint.Footprint.RowPitch,size_t(w)*8);
        D3D12_RANGE empty{};output.transfer->Unmap(0,&empty);
        if(actual!=expected){size_t i=0;while(actual[i]==expected[i])++i;
            throw std::runtime_error("GPU output differs at "+std::to_string(i)+" expected "+std::to_string(expected[i])+" actual "+std::to_string(actual[i]));}
    }
    require(gpu.errors()==0,"GPU codec debug errors");
    std::cout<<"GPU codec: exhaustive half/alpha, finite rounding midpoints, odd rows, motion/no-motion/reuse, debug layer passed\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
