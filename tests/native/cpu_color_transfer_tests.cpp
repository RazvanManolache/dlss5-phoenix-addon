// CPU-only numerical contracts for the legacy graphics/WOW64 pixel bridge.
#define NOMINMAX
#include "../../src/integration/compatibility/cpu_color_transfer.h"
#include "../../src/integration/compatibility/portable_policy.h"
#include <iostream>

int main() try {
    namespace color=phoenix_compat::cpu_color;
    unsigned checks=0;
    auto verify=[&](bool okay){++checks;if(!okay)throw std::runtime_error("CPU color transfer contract failed at check "+std::to_string(checks));};
    std::vector<uint8_t> bytes(256*4);
    for(size_t i=0;i<bytes.size();++i)bytes[i]=uint8_t(i);
    std::vector<uint16_t> packed(bytes.size());color::rgba8_to_half(bytes,packed);
    for(size_t i=0;i<bytes.size();++i)
        verify(packed[i]==DirectX::PackedVector::XMConvertFloatToHalf(bytes[i]/255.f));
    for(size_t count:{4u,12u,20u,28u}) {
        std::vector<uint8_t> tail(bytes.begin(),bytes.begin()+count);std::vector<uint16_t> result(count);
        color::rgba8_to_half(tail,result);
        for(size_t i=0;i<count;++i)verify(result[i]==color::byte_half_table()[tail[i]]);
    }
    std::vector<uint16_t> rgba(4,0);std::vector<float> rgb(3);
    for(unsigned h=0;h<65536;++h) {
        rgba[0]=rgba[1]=rgba[2]=uint16_t(h);rgba[3]=0x7e00; // Alpha is deliberately not an RGB validity guide.
        const bool finite=(h&0x7c00)!=0x7c00;
        bool threw=false;try{color::half_rgba_to_rgb(rgba,rgb);}catch(const std::runtime_error&){threw=true;}
        verify(threw!=finite);
        if(finite) {
            const auto expected=DirectX::PackedVector::XMConvertHalfToFloat(uint16_t(h));
            for(float value:rgb)verify(std::bit_cast<uint32_t>(value)==std::bit_cast<uint32_t>(expected));
            verify(color::finite(expected));
            verify(color::unorm8(expected)==uint8_t(std::lround(std::clamp(expected,0.f,1.f)*255.f)));
        }
        std::vector<uint8_t> original{1,2,3,73},packed_bytes;
        bool packed_threw=false;try{color::half_rgba_to_rgba8(rgba,original,packed_bytes);}catch(const std::runtime_error&){packed_threw=true;}
        verify(packed_threw!=finite);
        if(finite) {
            verify(packed_bytes[0]==color::unorm8(rgb[0]));
            verify(packed_bytes[1]==color::unorm8(rgb[1]));
            verify(packed_bytes[2]==color::unorm8(rgb[2]));
            verify(packed_bytes[3]==73);
        }
        std::vector<uint16_t> pair{uint16_t(h),uint16_t(h),uint16_t(h),0x7c01,uint16_t(h),uint16_t(h),uint16_t(h),0xfc00};
        std::vector<uint8_t> pair_input{1,2,3,73,4,5,6,19},pair_output;
        bool pair_threw=false;try{color::half_rgba_to_rgba8(pair,pair_input,pair_output);}catch(const std::runtime_error&){pair_threw=true;}
        verify(pair_threw!=finite);
        if(finite) {
            for(unsigned pixel=0;pixel<2;++pixel) {
                for(unsigned channel=0;channel<3;++channel)verify(pair_output[pixel*4+channel]==color::unorm8(rgb[channel]));
                verify(pair_output[pixel*4+3]==pair_input[pixel*4+3]);
            }
        }
    }
    // Exercise both adjacent floats around every quantization midpoint; adding
    // half in float precision (rather than double) fails some of these cases.
    for(unsigned byte=0;byte<255;++byte) {
        const float boundary=(float(byte)+.5f)/255.f;
        for(float value:{std::nextafter(boundary,0.f),boundary,std::nextafter(boundary,1.f)})
            verify(color::unorm8(value)==uint8_t(std::lround(value*255.f)));
    }
    for(float invalid:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        verify(!color::finite(invalid));bool threw=false;try{color::unorm8(invalid);}catch(const std::runtime_error&){threw=true;}verify(threw);
    }
    for(unsigned width:{1u,17u,64u}) {
        std::vector<uint8_t> input(size_t(width)*3*4);
        for(size_t i=0;i<input.size();++i)input[i]=uint8_t(i*73);
        auto enhanced=phoenix_compat::portable_color(input);
        for(size_t i=0;i<enhanced.size();++i)enhanced[i]=enhanced[i]*1.7f-.3f;
        phoenix_compat::Settings normal{};normal.mode=phoenix_addon::phoenix::ours;
        auto compare=normal;compare.mode=phoenix_addon::phoenix::split;
        compare.source_a=compare.source_b=2;compare.comparison=1;
        // Selecting the same enhanced image on both wipe sides must equal the
        // optimized enhanced-only path, including alpha and odd dimensions.
        verify(phoenix_compat::portable_compose(input,enhanced,width,normal)==
               phoenix_compat::portable_compose(input,enhanced,width,compare));
        normal.pipeline.compare_zoom=2;
        auto zoomed=phoenix_compat::portable_compose(input,enhanced,width,normal);
        for(size_t pixel=0;pixel<input.size()/4;++pixel) {
            auto xy=phoenix_pipeline::comparison_point(unsigned(pixel%width),unsigned(pixel/width),width,3,phoenix_pipeline::comparison_view_bits(normal.pipeline));
            for(unsigned c=0;c<3;++c)verify(zoomed[pixel*4+c]==color::unorm8(enhanced[(xy[1]*width+xy[0])*3+c]));
            verify(zoomed[pixel*4+3]==input[(xy[1]*width+xy[0])*4+3]);
        }
    }
    bool invalid_extent=false;try{std::vector<float> empty;color::half_rgba_to_rgb(rgba,empty);}catch(const std::runtime_error&){invalid_extent=true;}
    verify(invalid_extent);
    std::cout<<nlohmann::json{{"passed",true},{"checks",checks},{"cpu_only",true}}.dump()<<std::endl;
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
