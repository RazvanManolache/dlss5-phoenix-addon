#include "../../native_runtime/portable_pixels.h"
#include "../../native_runtime/frame_resolve.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cfenv>

int main() {
    phx::Kernel<HostMath> math{};
    // Every half code, all alpha bit patterns, odd row lengths, unaligned rows,
    // subnormals and nonfinite classification. Invalid colors remain invalid.
    std::vector<uint16_t> input(65536*4+1), rgba(65536*4), expected(rgba.size());
    std::vector<float> rgb(65536*3), motion(65536*2);
    for(size_t p=0;p<65536;++p) for(unsigned c=0;c<4;++c) input[1+p*4+c]=uint16_t(p+c);
    phoenix_portable_pixels::color(input.data()+1,rgb.data(),rgba.data(),65535);
    for(size_t p=0;p<65535;++p) {
        for(unsigned c=0;c<4;++c) if(rgba[p*4+c]!=input[1+p*4+c]) throw std::runtime_error("Alpha/input bits changed");
        for(unsigned c=0;c<3;++c) {
            const float value=std::clamp(math.half_value(input[1+p*4+c]),0.f,1.f),actual=rgb[p*3+c];
            if(std::isnan(value)?!std::isnan(actual):std::bit_cast<uint32_t>(value)!=std::bit_cast<uint32_t>(actual))
                throw std::runtime_error("Color conversion differs");
        }
    }
    phoenix_portable_pixels::motion(input.data()+1,motion.data(),65534,.375f,-.625f);
    for(size_t i=0;i<65534;++i) {
        const float value=math.half_value(input[i+1])*(i%2?-.625f:.375f);
        if(std::isnan(value)?!std::isnan(motion[i]):std::bit_cast<uint32_t>(value)!=std::bit_cast<uint32_t>(motion[i]))
            throw std::runtime_error("Motion scaling differs");
    }
    size_t count=0;
    // Include exact values and midpoints around every finite half boundary.
    for(unsigned h=0;h<0x7bff;++h) {
        float a=math.half_value(h),b=math.half_value(h+1);
        for(float value:{a,(a+b)*.5f,std::nextafter((a+b)*.5f,b),-a,-(a+b)*.5f}) {
            for(unsigned c=0;c<3;++c)rgb[count*3+c]=value;
            ++count;
            if(count==65535) {
                expected=rgba;
                for(size_t p=0;p<count;++p)for(unsigned c=0;c<3;++c)expected[p*4+c]=uint16_t(math.half_bits(rgb[p*3+c]));
                phoenix_portable_pixels::output(rgb.data(),rgba.data(),count);
                if(rgba!=expected)throw std::runtime_error("Output rounding/alpha differs");
                count=0;
            }
        }
    }
    expected=rgba;
    for(size_t p=0;p<count;++p)for(unsigned c=0;c<3;++c)expected[p*4+c]=uint16_t(math.half_bits(rgb[p*3+c]));
    phoenix_portable_pixels::output(rgb.data(),rgba.data(),count);
    if(rgba!=expected)throw std::runtime_error("Output tail differs");
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    if(phoenix_native_cuda::host_f16c_available()){
        uint32_t random=1947;
        auto value=[&]{random=random*1664525u+1013904223u;return float(random>>8)*0x1p-24f;};
        for(unsigned width:{1u,2u,3u,63u,257u,4097u}){
            std::vector<float> color(width*3),head(width*4),prior(width*3),actual(width*3);
            std::vector<uint8_t> valid(width);
            for(unsigned trial=0;trial<16;++trial){
                for(auto& v:color)v=value();for(auto& v:head)v=(value()-.5f)*8;
                for(auto& v:prior)v=value();for(auto& v:valid)v=value()>.5f;
                const float gain=value();
                // Unused gates may be NaN without affecting a valid RGB result.
                for(unsigned p=0;p<width;++p)if(!valid[p])head[p*4+3]=std::bit_cast<float>(0x7f800001u);
                std::feclearexcept(FE_INVALID);
                phoenix_cpp::resolve_neural_row(color.data(),head.data(),prior.data(),valid.data(),gain,actual.data(),width);
                if(std::fetestexcept(FE_INVALID))throw std::runtime_error("Unused history logit raised an FP exception");
                for(unsigned p=0;p<width;++p)for(unsigned c=0;c<3;++c){
                    float network=std::clamp(color[p*3+c]+head[p*4+c]*.25f,0.f,1.f);
                    network+=(prior[p*3+c]-network)*phoenix_cpp::resolve_gate(head[p*4+3],valid[p]!=0,gain);
                    const float reference=math.half(network);
                    if(std::bit_cast<uint32_t>(reference)!=std::bit_cast<uint32_t>(actual[p*3+c]))
                        throw std::runtime_error("Vector temporal resolve differs from scalar contract");
                }
            }
        }
    }
#endif
    std::cout<<"Exhaustive half decode, scaled motion, finite midpoint rounding and alpha passed; F16C="
        <<phoenix_native_cuda::host_f16c_available()<<"; vector temporal resolve exact\n";
}
